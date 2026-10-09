"""Volume-enabled GRPO runner, without changing the supplied modal_utils."""

import subprocess
import sys

import modal

from cs336_alignment.modal_utils import (
    GPU, MAX_CONTAINERS, RUN_TIMEOUT_SECONDS, SUNET_ID,
    app, image, wandb_secret,
)

VOLUME_NAME = f"cs336-a5-grpo-checkpoints-{SUNET_ID}"
VOLUME_MOUNT = "/mnt/checkpoints"
checkpoint_volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    secrets=[wandb_secret],
    volumes={VOLUME_MOUNT: checkpoint_volume},
)
def run_training(training_args: list[str]) -> str:
    # A warm container may have an older mounted snapshot from its previous job.
    checkpoint_volume.reload()
    # Keep the original subprocess isolation for CUDA/NCCL and vLLM lifecycle.
    # The child inherits Modal's container credentials and mounted Volume.
    entrypoint = (
        "import sys, modal; "
        "from scripts.train_grpo import main; "
        "volume = modal.Volume.from_name(sys.argv[1]); "
        "main(sys.argv[2:], checkpoint_commit=volume.commit)"
    )
    subprocess.run(
        [sys.executable, "-u", "-c", entrypoint, VOLUME_NAME, *training_args],
        check=True,
    )
    return " ".join(training_args)


def submit_training(jobs: list[list[str]]) -> None:
    print(f"Submitting {len(jobs)} GRPO jobs with {GPU}, Volume={VOLUME_NAME}", flush=True)
    failures = []
    for index, result in enumerate(run_training.map(jobs, return_exceptions=True)):
        args = jobs[index]
        if isinstance(result, BaseException):
            print(f"Failed: {args}\n{result!r}", flush=True)
            failures.append(args)
        else:
            print(f"Completed: {result}", flush=True)
    if failures:
        raise SystemExit(1)
