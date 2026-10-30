"""
Zero-shot SimpleSafetyTests baseline - Evaluate Qwen2.5-7B zero-shot performance on SimpleSafetyTests.
-> output Model generations
Running:

```
python scripts/evaluate_alpaca_baseline.py \
    --input-path <path_to_alpaca_eval> "data/simple_safety_tests/simple_safety_tests.csv" \
    --prompt-path <path_to_prompt> "cs336_alignment/prompts_safety/simple_safety_tests_zero_shot.prompt" \
    --model-name-or-path "/mnt/cs336-a5-supplement/models/Qwen2.5-7B" \
    --num-gpus 1 \
    --output-path <path_to_write_output.jsonl> "/mnt/cs336-a5-supplement-results/simple_safety_tests_qwen2_5_7B.jsonl"
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
from cs336_alignment.Safety_RLHF.zero_shot_evaluation import *

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
    input_examples:List[Dict[str, Any]] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)          # 第一行当表头，按列名取值
        for row in reader:
            input_examples.append({
                "instruction": row["prompts_final"],   # prompts_final -> instruction
            })
    logger.info(f"Read {len(input_examples)} simple_safety_tests examples from {input_path}")
    
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
    with open(output_path, "w", encoding="utf-8") as fout:
        for example, response in tqdm(
            zip(input_examples, responses),
            total=len(input_examples),
            desc="Serializing SimpleSafetyTests predictions",
        ):
            obj = {
                "prompts_final": example["instruction"],   # 来自 SimpleSafetyTests 的指令
                "output": response,                          # 模型对该指令的输出
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/simple_safety_tests/simple_safety_tests.csv"
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="cs336_alignment/prompts_safety/simple_safety_tests_zero_shot.prompt"
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
        default="/mnt/cs336-a5-supplement-results/simple_safety_tests_qwen2_5_7B.jsonl"
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