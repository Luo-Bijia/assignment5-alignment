from cs336_alignment.modal_utils_safety import app, submit_commands

# 全量评估 805 组回答比较

@app.local_entrypoint()
def main() -> None:
    command = [
        "alpaca_eval",
        "--model_outputs",
        "data/alpaca_eval/alpaca_eval_qwen2_5_7B.json",
        "--reference_outputs",
        "data/alpaca_eval/alpaca_eval_gpt4_turbo.json",
        "--annotators_config",
        "scripts/alpaca_eval_vllm_qwen2_5_72b_fn",
        "--output_path",
        "/mnt/cs336-a5-supplement-results/alpaca_eval_qwen2_5_72b_judge",
        "--base-dir",
        ".",
    ]

    submit_commands([command])