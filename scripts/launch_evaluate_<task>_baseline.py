from cs336_alignment.modal_utils_safety import (
    BASE_MODEL_PATH,
    RESULTS_VOLUME_MOUNT_PATH,
    app,
    submit_commands,
)


@app.local_entrypoint()
def main(*argv: str) -> None:
    command = [
        "python", "-u", "-m", "scripts.evaluate_safety",
        *argv,
    ]
    submit_commands([command])
