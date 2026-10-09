import argparse

from cs336_alignment.modal_checkpoint_utils import VOLUME_MOUNT, app, submit_training

"""训练 GRPO 的 Modal 提交入口代码"""
@app.local_entrypoint()
def main(*argv: str) -> None:
    parser = argparse.ArgumentParser(
        description="Submit GRPO training jobs to Modal",
        allow_abbrev=False,
    )
    parser.add_argument("--seeds", default="0")
    args, training_args = parser.parse_known_args(list(argv))

    seeds = [
        str(int(seed.strip()))
        for seed in args.seeds.split(",")
        if seed.strip()
    ]
    if not seeds:
        parser.error("--seeds must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        parser.error("Duplicate seeds would write to the same checkpoint directory")
    if any(arg == "--seed" or arg.startswith("--seed=")
           for arg in training_args):
        parser.error("Use --seeds to select seeds, not --seed")

    # Only supply defaults when the caller has not selected another directory.
    for flag, default in (
        ("--checkpoint-dir", f"{VOLUME_MOUNT}/checkpoints"),
        ("--output-dir", f"{VOLUME_MOUNT}/results"),
    ):
        if not any(arg == flag or arg.startswith(f"{flag}=") for arg in training_args):
            training_args.extend([flag, default])

    jobs = [
        [
            "--seed", seed,
            *training_args,
        ]
        for seed in seeds       # 传单个0就是只跑一个双卡的训练，传0、1、2、3就是四个并行独立的双卡训练（八卡额度，衡量结果的随机波动）
    ]
    submit_training(jobs)
