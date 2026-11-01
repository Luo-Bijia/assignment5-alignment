"""
Zero-shot AlpacaEval baseline - Evaluate Qwen2.5-7B zero-shot performance on AlpacaEval.
-> output Model generations
Running:

```
python scripts/evaluate_alpaca_baseline.py \
    --input-path <path_to_alpaca_eval> "data/alpaca_eval/alpaca_eval_gpt4_turbo.json" \
    --prompt-path <path_to_prompt> "cs336_alignment/prompts_safety/alpaca_eval_zero_shot.prompt" \
    --model-name-or-path "/mnt/cs336-a5-supplement/models/Qwen2.5-7B" \
    --num-gpus 1 \
    --output-path <path_to_write_output.jsonl> "/mnt/cs336-a5-supplement-results/alpaca_eval_qwen2_5_7B.json"
```
"""

import csv
import os
import argparse
import json
import logging
from pathlib import Path
import sys
from statistics import mean
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen
from cs336_alignment.supplement.zero_shot_evaluation import *

logger = logging.getLogger(__name__)

def main(input_path:str | os.PathLike,
         prompt_path:str | os.PathLike,
         model_name_or_path:str | os.PathLike,
         num_gpus:int,
         output_path:str | os.PathLike):
    
    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    
    # 1.load examples
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)          # 整个文件是一个 JSON 数组
    input_examples:List[Dict[str, Any]] = []
    for record in records:
        input_examples.append({
            "dataset": record["dataset"],
            "instruction": record["instruction"],
        })
    logger.info(f"Read {len(input_examples)} alpaca_eval examples from {input_path}")
    
    # 2.format prompts
    sys_prompt_path = "cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt"
    task_template = Path(prompt_path).read_text(encoding='utf-8')
    sys_template = Path(sys_prompt_path).read_text(encoding='utf-8')
    prompts:List[str] = [sys_template.format(instruction=
        task_template.format(instruction=p["instruction"]))
        for p in input_examples]
    
    # 3.generate outputs
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
    raw_responses = model.generate(prompts, sampling_params)     # 所有 examples 一次喂入
    responses:List[str] = []
    for output in raw_responses:
        response = output.outputs[0].text.strip()   # .text：生成的文本字符串（只有新生成部分，不含 prompt）
        responses.append(response)
    assert len(responses) == len(prompts)
    logger.info(f"Processed {len(prompts)} prompts")
    
    # 4.serialize the outputs and model generations to disk
    generations = []
    for example, response in tqdm(
        zip(input_examples, responses),
        total=len(input_examples),
        desc="Serializing generations to disk",
    ):
        generations.append({
            "dataset": example["dataset"],
            "instruction": example["instruction"],
            "output": response,
            "generator": "Qwen2.5-7B",
        })

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(generations, fout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/alpaca_eval/alpaca_eval_gpt4_turbo.json"
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="cs336_alignment/prompts_safety/alpaca_eval_zero_shot.prompt"
    )
    parser.add_argument(
        "--model-name-or-path", help="HF name of the model to use",
        type=str,
        default="/mnt/cs336-a5-supplement/models/Qwen2.5-7B"
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to write output metrics",
        default="/mnt/cs336-a5-supplement-results/alpaca_eval_qwen2_5_7B.json"
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.prompt_path,
        args.model_name_or_path,
        args.num_gpus,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])