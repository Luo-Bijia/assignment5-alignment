import math
from typing import Callable, Iterable, Literal

import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer
from cs336_alignment.sft_utils import * 

def compute_rollout_rewards(reward_fn: Callable[[str, str], dict[str, float]], 
                            rollout_responses: list[str], 
                            repeated_ground_truths: list[str]
                            ) -> tuple[torch.Tensor, dict[str, float]]:
    # Compute rewards for a list of rollout responses, along with metadata for the reward components.
    assert len(rollout_responses) == len(repeated_ground_truths)
    rollout_batch_size = len(rollout_responses)
    total_rewards, format_rewards, answer_rewards = 0.0, 0.0, 0.0
    raw_rewards = []

    for rollout, gt in zip(rollout_responses, repeated_ground_truths):
        res = reward_fn(rollout, gt)
        raw_rewards.append(res["reward"])
        total_rewards += res["reward"]
        format_rewards += res["format_reward"]
        answer_rewards += res["answer_reward"]
    raw_rewards = torch.tensor(raw_rewards)

    return (raw_rewards, # (rollout_batch_size,)
            {"mean_total_rewards": total_rewards / rollout_batch_size,
             "mean_format_rewards": format_rewards / rollout_batch_size,
             "mean_answer_rewards": answer_rewards / rollout_batch_size})

def compute_group_normalized_rewards(raw_rewards: torch.Tensor, 
                                     group_size: int, 
                                     baseline: Literal["mean", "none"] = "mean", 
                                     advantage_eps: float = 1e-6, 
                                     advantage_normalizer: Literal["std", "none", "mean"] = "std"
                                     ) -> tuple[torch.Tensor, dict[str, float]]:
    # Compute advantages by applying the requested baseline and normalization within each group.
    # (raw_rewards - mean) / std at G（组内） -> advantage
    advantages = torch.zeros_like(raw_rewards, device=raw_rewards.device)
    group_start = 0
    # 枚举每个同 prompt 下的 group
    while group_start < len(raw_rewards):
        group_b = 0.0
        if baseline == "mean":
            group_b = torch.mean(raw_rewards[group_start : group_start + group_size]).item()

        group_norm = 1.0
        if advantage_normalizer == "std":
            group_norm = advantage_eps + torch.std(raw_rewards[group_start : group_start + group_size]).item()
        elif advantage_normalizer == "mean":
            group_norm = advantage_eps + torch.mean(raw_rewards[group_start : group_start + group_size]).item()

        advantages[group_start : group_start + group_size] = (raw_rewards[group_start : group_start + group_size] - group_b) / group_norm

        group_start += group_size

    return (advantages, # (rollout_batch_size,)
            {"mean_rewards" : torch.mean(raw_rewards),
             "std_rewards": torch.std(raw_rewards),
             "max_reward": torch.max(raw_rewards),
             "min_reward": torch.min(raw_rewards)
            })

def compute_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor, 
                                 policy_log_probs: torch.Tensor, 
                                 importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none", 
                                 old_log_probs: torch.Tensor | None = None, 
                                 cliprange: float | None = None, 
                                 response_mask: torch.Tensor | None = None
                                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    A = raw_rewards_or_advantages.unsqueeze(1)
    tokens_loss = - A * policy_log_probs        # = -goal
    return (tokens_loss, # (batch_size, sequence_length)
            { }
            )

def aggregate_loss_across_microbatch(per_token_policy_gradient_loss: torch.Tensor, 
                                     mask: torch.Tensor, 
                                     loss_normalization: Literal["sequence", "constant"] = "sequence", 
                                     normalization_constant: int | None = None, ) -> torch.Tensor:
    # batch_size = B*G
    responses_len = torch.sum(mask, dim=1, keepdim=True)
    # 长度归一化
    if loss_normalization == "sequence":
        per_token_policy_gradient_loss /= responses_len
    else:
        assert normalization_constant is not None
        per_token_policy_gradient_loss /= normalization_constant

    responses_gradient_loss = per_token_policy_gradient_loss * mask     # extract response

    loss = torch.mean( torch.sum(responses_gradient_loss, dim=1) )       # 在这个microbatch下整体平均：1/BG * Σ
    return loss     # scalar

def grpo_train_step(model: PreTrainedModel, 
                    tokenizer: PreTrainedTokenizer, 
                    optimizer: Optimizer, 
                    gradient_accumulation_steps: int, 
                    max_grad_norm: float | None, 
                    reward_fn: Callable[[str, str], dict[str, float]], 
                    # train data
                    repeated_prompts: list[str], 
                    rollout_responses: list[str], 
                    repeated_ground_truths: list[str], 
                    group_size: int, 
                    # Reward normalization 
                    baseline: Literal["mean", "none"] = "mean", 
                    advantage_eps: float = 1e-6, 
                    advantage_normalizer: Literal["std", "none", "mean"] = "std", 
                    # Importance reweighting and clipping 
                    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none", 
                    old_log_probs: torch.Tensor | None = None, 
                    cliprange: float | None = None, 
                    # Loss normalization 
                    loss_normalization: Literal["sequence", "constant"] = "sequence", 
                    normalization_constant: int | None = None
                    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    # 一个 grpo_train_step = 一个 rollout batch 上的一次 optimizer update
    # 把这一次 update 拆成 gradient_accumulation_steps 次较小的 forward/backward，但最后只调用一次 optimizer.step()。

    # 对整个 rollout batch 计算 rewards，按每个 group 的 G 条回答计算 advantages
    raw_rewards, metadata_rewards = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, metadata_gn_rewards = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)

    # tokenize 整个 rollout batch
    result1 = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = result1["input_ids"], result1["labels"], result1["response_mask"]

    # CUDA device 对齐
    device = next(model.parameters()).device    # 拿模型第一个参数所在的设备，作为整个模型所在的设备。
    input_ids, labels, response_mask, advantages = input_ids.to(device), labels.to(device), response_mask.to(device), advantages.to(device)

    step_loss = 0.0
    microbatch_size = len(input_ids) // gradient_accumulation_steps
    # 跨 microbatch 汇总 token entropy
    total_response_tokens = 0
    total_token_entropy = 0.0
    for i in range(0, len(input_ids), microbatch_size):
        input_ids_microbatch, labels_microbatch, advantages_microbatch, response_mask_microbatch = input_ids[i : i+microbatch_size], labels[i : i+microbatch_size], advantages[i : i+microbatch_size], response_mask[i : i+microbatch_size, :]

        # Forward pass
        result2 = get_response_log_probs(model, input_ids_microbatch, labels_microbatch, True)
        log_probs_microbatch, token_entropy_microbatch = result2["log_probs"], result2["token_entropy"]
        per_token_policy_gradient_loss_microbatch, _ = compute_policy_gradient_loss(advantages_microbatch, log_probs_microbatch, importance_reweighting_method, old_log_probs, cliprange, response_mask_microbatch)


        loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss_microbatch, response_mask_microbatch, loss_normalization, normalization_constant)
        # 乘 microbatch 比例，确保与一次性把整个 batch 放进模型得到的梯度相同，只是显存占用更小
        loss *= len(input_ids_microbatch) / len(input_ids)

        # 累计 response token 的 entropy（只统计 response，排除 prompt/padding）
        # token_entropy_microbatch 形状应为 (batch, seq_len)，与 response_mask 对应
        entropy_masked = token_entropy_microbatch * response_mask_microbatch
        total_token_entropy += entropy_masked.sum().item()
        total_response_tokens += response_mask_microbatch.sum().item()

        # Backward pass
        loss.backward()
        step_loss += loss.item()

    metadata = {}

    metadata["loss"] = step_loss

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += torch.sum(p.grad**2).item()
    metadata["grad_norm"] = math.sqrt(grad_norm)

    if max_grad_norm is not None:
        clip_grad_norm_(model.parameters(), max_grad_norm)

    # token entropy: 跨所有 microbatch 的 response token 平均
    metadata["token_entropy"] = (
        total_token_entropy / total_response_tokens if total_response_tokens > 0 else 0.0
    )
    # train rewards
    metadata["train_total_reward"] = metadata_rewards["mean_total_rewards"]
    metadata["train_format_reward"] = metadata_rewards["mean_format_rewards"]
    metadata["train_answer_reward"] = metadata_rewards["mean_answer_rewards"]
    # 保留 group normalized rewards 的统计
    metadata["mean_rewards"] = metadata_gn_rewards["mean_rewards"]
    metadata["std_rewards"] = metadata_gn_rewards["std_rewards"]

    # Update weights once across entire batch. 
    optimizer.step() 
    # Zero gradients once across entire batch. 
    optimizer.zero_grad()
    return (torch.tensor(step_loss), metadata)