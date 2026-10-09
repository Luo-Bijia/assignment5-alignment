import argparse
import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import torch
import wandb
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from wandb.sdk.wandb_run import Run

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.GRPO import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.training_checkpoint import (
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)


class GSM8KExample(TypedDict):
    question: str
    ground_truth: str


class RolloutRow(TypedDict):
    question: str
    response: str
    ground_truth: str
    length: int
    finish_reason: str | None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument(
        "--prompt",
        default="cs336_alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument("--train-path", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val-path", default="data/gsm8k/test.jsonl")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--val-temperature", type=float, default=0.0)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--n-log-examples", type=int, default=16)

    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8000)

    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-group", default="standard-on-policy")
    parser.add_argument("--wandb-mode", choices=["online", "offline"],
                        default="online")
    parser.add_argument("--output-dir", default="experiments/grpo")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--checkpoint-dir", help="Checkpoint root; each seed has its own subdirectory")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser


def load_gsm8k(path: str | Path, count: int) -> list[GSM8KExample]:
    """
    读取 GSM8K数据集，返回 [{"question": ..., "ground_truth": ...}, ...]，取前 count 条。
    """
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if "####" not in item["answer"]:
                raise ValueError(f"Missing final-answer delimiter in {path}")
            examples.append({
                "question": item["question"],
                "ground_truth": item["answer"].rsplit("####", 1)[1].strip(),
            })

    if len(examples) < count:
        raise ValueError(
            f"{path} has {len(examples)} examples, requested {count}"
        )
    return examples[:count]


def generate(
    server: VLLMServer,
    examples: list[GSM8KExample],
    template: str,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    n: int,
    seed: int,
    temperature: float,
) -> tuple[list[str], list[str], list[str], list[RolloutRow]]:
    # 1. 把每个 question 填进 r1_zero 模板。模板里用 {question} 占位。
    prompts = [
        template.format(question=item["question"])
        for item in examples
    ]
    # 2. 调 vLLM 生成。每题 n 条，共 len(examples)*n 条
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": temperature,
            "max_tokens": args.max_tokens,
            "n": n,
            "seed": seed,
            "stop": ["</answer>"],      # 遇到 </answer> 停止
            "include_stop_str_in_output": True,      # 把 </answer> 保留在输出里，供 grader 判格式
        },
        batch_size=args.generation_batch_size,      # 分批发 HTTP，避免单请求过大
    )

    # 3. vLLM 返回的是扁平列表，长度必须是 prompts * n
    if len(completions) != len(examples) * n:
        raise RuntimeError("Unexpected number of vLLM completions")

    # 4. 把 prompt / example 各重复 n 次，与 completions 逐项对齐
    #    顺序假设：vLLM 按 prompt0 的 n 条 → prompt1 的 n 条 → … 返回
    repeated_prompts = [
        prompt for prompt in prompts for _ in range(n)
    ]
    repeated_examples = [
        item for item in examples for _ in range(n)
    ]
    responses = [completion.text for completion in completions]

    # 5. 构造用于记录/统计的 rows，并做空回答检查
    rows = []
    for item, completion in zip(repeated_examples, completions):
        # 空回答会导致 sequence loss 除以 0，必须提前拦截
        if not tokenizer.encode(completion.text, add_special_tokens=False):
            raise RuntimeError("Empty response: sequence loss would divide by zero")

        # 优先用 vLLM 返回的真实 token id 长度；缺失时回退到重新 tokenize
        length = len(completion.token_ids) if completion.token_ids else len(
            tokenizer.encode(completion.text, add_special_tokens=False)
        )
        rows.append({
            "question": item["question"],
            "response": completion.text,
            "ground_truth": item["ground_truth"],
            "length": length,
            "finish_reason": completion.finish_reason,      # 'stop' / 'length'
        })

    # 6. ground truth 也按同样顺序重复 n 次，与 responses 对齐
    ground_truths = [item["ground_truth"] for item in repeated_examples]
    return repeated_prompts, responses, ground_truths, rows


def log_rollouts(
    run: Run,
    rows: list[RolloutRow],
    split: str,
    step: int,
    output_dir: Path,
    count: int,
) -> None:
    selected = rows[:count]     # 只记录前 count 条，避免 W&B 表格过大
    columns = [
        "step", "question", "response", "ground_truth",
        "reward", "format_reward", "answer_reward",
        "length", "finish_reason",
    ]
    records = []
    table = wandb.Table(columns=columns)
    for row in selected:
        # 把 reward 拆解成三个分量一起记录，便于分析涨的是格式还是答案
        record = {
            "step": step,
            **row,
            **r1_zero_reward_fn(row["response"], row["ground_truth"]),
        }
        records.append(record)
        table.add_data(*(record[column] for column in columns))

    # 双通道：W&B 表格（在线看）+ 本地 json（artifact 上传/离线分析）
    run.log({
        "optimizer_step": step,
        f"{split}/rollouts_step_{step}": table,
    })
    path = output_dir / f"{split}_rollouts_{step:04d}.json"
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def evaluate(
    server: VLLMServer,
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[GSM8KExample],
    template: str,
    args: argparse.Namespace,
    run: Run,
    step: int,
    output_dir: Path,
) -> dict[str, float]:
    # 验证必须看到最新 policy：先把权重同步到 vLLM
    server.sync_policy_weights(policy)

    # 每题只生成 1 条，贪心（val_temperature=0.0），保证可复现
    _, responses, ground_truths, rows = generate(
        server, examples, template, tokenizer, args,
        n=1,
        seed=args.seed,
        temperature=args.val_temperature,
    )
    scores = [
        r1_zero_reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]
    metrics = {
        f"val/{key}": sum(score[key] for score in scores) / len(scores)
        for key in ("reward", "format_reward", "answer_reward")
    }
    metrics["val/avg_response_length"] = (
        sum(row["length"] for row in rows) / len(rows)
    )
    # 截断率：多少回答因为达到 max_tokens 被截断，反映生成长度是否够
    metrics["val/truncation_fraction"] = (
        sum(row["finish_reason"] == "length" for row in rows) / len(rows)
    )
    run.log({"optimizer_step": step, **metrics})
    log_rollouts(
        run, rows, "val", step, output_dir, args.n_log_examples
    )
    print(json.dumps({"step": step, **metrics}), flush=True)
    return metrics


def main(
    argv: list[str] | None = None,
    checkpoint_commit: Callable[[], None] | None = None,
) -> None:
    args = make_parser().parse_args(argv)
    
    # 参数校验
    positive = [
        args.n_train_examples, args.n_val_examples,
        args.num_rollout_steps, args.rollout_batch_size,
        args.gradient_accumulation_steps, args.generation_batch_size,
        args.max_tokens, args.eval_every, args.log_rollouts_every,
        args.n_log_examples,
        args.checkpoint_every,
    ]
    if min(positive) <= 0:
        raise ValueError("Counts and intervals must be positive")
    if args.group_size < 2:
        # 组内只有 1 条时 std=0、advantage 无意义，GRPO 退化
        raise ValueError("Standard GRPO requires group_size >= 2")
    if args.rollout_batch_size % args.group_size:
        # 必须整除，才能把 rollout batch 整齐切成 group
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if args.rollout_batch_size % args.gradient_accumulation_steps:
        # 必须整除，microbatch 大小才均匀
        raise ValueError("Batch must be divisible by accumulation steps")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires NVIDIA CUDA GPUs")

    # 设备校验
    policy_device = torch.device(args.policy_device)
    if policy_device.type != "cuda" or policy_device.index is None:
        # 必须显式写 cuda:0，不能只写 cuda
        raise ValueError("Use an explicit policy device, e.g. cuda:0")
    if policy_device.index == args.vllm_gpu:
        # NCCL 权重同步要求训练侧和推理侧在不同 GPU，否则会冲突
        raise ValueError("The existing NCCL helper requires separate GPUs")
    if max(policy_device.index, args.vllm_gpu) >= torch.cuda.device_count():
        raise ValueError("Requested GPU index is not visible")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    train_examples = load_gsm8k(args.train_path, args.n_train_examples)
    val_examples = load_gsm8k(args.val_path, args.n_val_examples)
    rng.shuffle(train_examples)
    template = Path(args.prompt).read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError("Prompt template must contain {question}")

    n_prompts = args.rollout_batch_size // args.group_size      # 256/8 = 32
    if len(train_examples) < n_prompts:
        raise ValueError("Not enough training questions for one rollout batch")

    checkpoint_dir = (
        Path(args.checkpoint_dir) / f"seed-{args.seed}"
        if args.checkpoint_dir else None
    )
    # Operational flags may change on resume; training/data settings may not.
    resume_keys = (
        "model", "seed", "n_train_examples", "n_val_examples",
        "rollout_batch_size", "group_size", "gradient_accumulation_steps",
        "learning_rate", "max_grad_norm", "temperature", "max_tokens",
        "val_temperature", "generation_batch_size", "wandb_project", "wandb_mode",
    )
    checkpoint_config = {key: getattr(args, key) for key in resume_keys}
    if checkpoint_dir is not None:
        for key in ("train_path", "val_path", "prompt"):
            checkpoint_config[f"{key}_sha256"] = hashlib.sha256(
                Path(getattr(args, key)).read_bytes()
            ).hexdigest()
    checkpoint = load_training_checkpoint(checkpoint_dir) if checkpoint_dir else None
    if checkpoint is not None:
        if checkpoint["config"] != checkpoint_config:
            changed = [key for key in checkpoint_config
                       if checkpoint["config"].get(key) != checkpoint_config[key]]
            raise ValueError(f"Checkpoint configuration changed: {changed}; use a new --checkpoint-dir")
        if args.num_rollout_steps < checkpoint["step"]:
            raise ValueError("num_rollout_steps is the TOTAL target, below the saved step")

    # 输出
    output_dir = Path(args.output_dir) / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )   # 先落盘配置，即使后面崩溃也能知道跑了什么

    server = VLLMServer(
        model_id=args.model,
        gpu=args.vllm_gpu,
        port=args.vllm_port,
        seed=args.seed,
    )
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,     # 4 seed 同 group，便于对齐比较
        name=f"grpo-seed-{args.seed}",
        config=vars(args),
        mode=args.wandb_mode,
        allow_val_change=checkpoint is not None,
        **({"id": checkpoint["run_id"], "resume": "allow"}
           if checkpoint is not None and args.wandb_mode == "online" else {}),
    )
    # 自定义横轴为 optimizer_step，而不是 W&B 默认的 log 次数
    run.define_metric("optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("val/*", step_metric="optimizer_step")

    # 训练主体
    completed = False
    try:
        # 1. 加载 policy + tokenizer 到 GPU 0
        policy, tokenizer = get_model_and_tokenizer(
            args.model, args.policy_device
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token   # 很多模型没有 pad token
        tokenizer.padding_side = "right"    # 右侧 padding，配合 causal LM
        policy.config.use_cache = False     # 训练时禁用 KV cache，避免和梯度冲突

        # eval() 关闭 dropout，但不会关闭梯度，autograd 仍正常
        policy.eval()

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        optimizer.zero_grad(set_to_none=True)       # set_to_none 更省显存

        # 2. 启动 vLLM + 建立 NCCL 权重同步通道
        server.start()
        server.init_weight_sync(args.policy_device)

        start_step = 0
        cursor = 0
        final_metrics = None
        if checkpoint is not None:
            restore_training_checkpoint(checkpoint, policy, optimizer, rng)
            start_step = checkpoint["step"]
            cursor = checkpoint["cursor"]
            train_examples = checkpoint["train_examples"]
            final_metrics = checkpoint["final_metrics"]
            print(f"Resuming after step {start_step}; next step is {start_step + 1}", flush=True)
        else:
            # 3. 只有新训练才记录 step=0。
            evaluate(
                server, policy, tokenizer, val_examples, template,
                args, run, 0, output_dir,
            )

        for step in range(start_step + 1, args.num_rollout_steps + 1):
            # 4. 取 32 道训练题；用尽则重新 shuffle 从头取（数据循环）
            if cursor + n_prompts > len(train_examples):
                rng.shuffle(train_examples)
                cursor = 0
            examples = train_examples[cursor:cursor + n_prompts]
            cursor += n_prompts

            # 5. 同步最新权重到 vLLM，保证 on-policy
            server.sync_policy_weights(policy)

            # 6. 每题生成 group_size 条回答
            prompts, responses, ground_truths, rows = generate(
                server, examples, template, tokenizer, args,
                n=args.group_size,
                seed=args.seed + step,      # 每步不同 seed，增加采样多样性
                temperature=args.temperature,
            )

            # 7. 一次 GRPO 更新
            loss, metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=prompts,
                rollout_responses=responses,
                repeated_ground_truths=ground_truths,
                group_size=args.group_size,
                baseline="mean",
                advantage_normalizer="std",
                importance_reweighting_method="none",       # 标准 on-policy GRPO
                loss_normalization="sequence",
            )

            # 8. 整理标量日志，检查有限性
            scalars = {
                key: float(value.detach().cpu())
                if isinstance(value, torch.Tensor) else float(value)
                for key, value in metadata.items()
            }
            scalars["loss"] = float(loss.detach().cpu())
            scalars["avg_response_length"] = (
                sum(row["length"] for row in rows) / len(rows)
            )
            if any(not torch.isfinite(torch.tensor(v)) for v in scalars.values()):
                # NaN/Inf 早停，避免污染后续所有步骤
                raise RuntimeError(f"Non-finite training metric at step {step}")

            run.log({
                "optimizer_step": step,
                **{f"train/{key}": value for key, value in scalars.items()},
            })
            print(
                f"step={step} loss={scalars['loss']:.4f} "
                f"reward={scalars['train_total_reward']:.4f}",
                flush=True,
            )

            # 9. 第 1 步 + 每 log_rollouts_every 步记录 rollout 示例
            if step == 1 or step % args.log_rollouts_every == 0:
                log_rollouts(
                    run, rows, "train", step, output_dir,
                    args.n_log_examples,
                )

            # 10. 每 eval_every 步（及最后一步）验证
            if step % args.eval_every == 0 or step == args.num_rollout_steps:
                final_metrics = evaluate(
                    server, policy, tokenizer, val_examples, template,
                    args, run, step, output_dir,
                )

            if checkpoint_dir is not None and (
                step % args.checkpoint_every == 0 or step == args.num_rollout_steps
            ):
                save_training_checkpoint(
                    checkpoint_dir, policy, optimizer, step, cursor,
                    train_examples, rng, checkpoint_config, run.id,
                    final_metrics, checkpoint_commit,
                )

        # A completed checkpoint can be re-submitted to finish export/logging.
        if start_step == args.num_rollout_steps or final_metrics is None:
            final_metrics = evaluate(
                server, policy, tokenizer, val_examples, template,
                args, run, args.num_rollout_steps, output_dir,
            )

        run.summary["final_val_accuracy"] = final_metrics["val/reward"]
        (output_dir / "final_metrics.json").write_text(
            json.dumps(final_metrics, indent=2), encoding="utf-8"
        )

        if args.save_model:
            policy.save_pretrained(output_dir / "model")
            tokenizer.save_pretrained(output_dir / "model")

        if checkpoint_commit is not None:
            checkpoint_commit()

        # 保留原有 artifact 导出；checkpoint 位于独立目录，不随结果上传。
        artifact = wandb.Artifact(
            name=f"grpo-results-{run.id}",
            type="grpo-results",
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
        completed = True
    finally:
        try:
            server.stop()       # 无论如何都要关 vLLM，释放 GPU
        finally:
            run.finish(exit_code=0 if completed else 1)     # 失败时标记 run 为 crashed


if __name__ == "__main__":
    main()
