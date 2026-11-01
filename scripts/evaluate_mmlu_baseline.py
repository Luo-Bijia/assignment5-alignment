"""
Zero-shot MMLU baseline - Evaluate Qwen2.5-7B zero-shot performance on MMLU.

Running:

```
python scripts/evaluate_mmlu_baseline.py \
    --input-path <directory_to_mmlu> "data/mmlu/test" \
    --prompt-path <path_to_prompt> "cs336_alignment/prompts_safety/mmlu_zero_shot.prompt" \
    --model-name-or-path "/mnt/cs336-a5-supplement/models/Qwen2.5-7B" \
    --num-gpus 2 \
    --output-path <path_to_write_output.jsonl> "/mnt/cs336-a5-supplement-results/mmlu_zero_shot_metrics.jsonl"
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
    input_examples:List[Dict[str, Any]] = []
    csv_paths = [str(p) for p in Path(input_path).glob("*.csv")]    # input_path目录下的57个学科的csv文件
    
    # 1.load examples
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)                # 无表头，逐行读
            subject = csv_path.split('.')[0]
            subject = subject[:-len("_test")] if "_test" in subject else subject[:-len("_dev")]     # remove suffix
            subject = subject.split('/')[-1]
            for row in reader:
                if len(row) < 6:                  # 跳过空行/异常行
                    continue
                question = row[0]
                options = [row[1], row[2], row[3], row[4]]
                answer = row[5].strip().upper()   # 确保是 "A"/"B"/"C"/"D"

                input_examples.append({
                    "subject": subject,
                    "question": question,
                    "options": options,
                    "answer": answer,
                })
    logger.info(f"Read {len(input_examples)} mmlu examples from {input_path}")
    
    # 2.format prompts
    sys_prompt_path = "cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt"
    task_template = Path(prompt_path).read_text(encoding='utf-8')
    sys_template = Path(sys_prompt_path).read_text(encoding='utf-8')
    prompts:List[str] = [sys_template.format(instruction=
        task_template.format(subject=p["subject"], question=p["question"], options=p["options"]))
        for p in input_examples]
    
    # 3.generate outputs
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, stop=["\n# Query:", "# Query:"])        # stop 根据结果呈现规律来定的
    raw_responses = model.generate(prompts, sampling_params)     # 所有 examples 一次喂入
    responses:List[str] = []
    for output in raw_responses:
        response = output.outputs[0].text.strip()   # .text：生成的文本字符串（只有新生成部分，不含 prompt）
        responses.append(response)
    assert len(responses) == len(prompts)
    logger.info(f"Processed {len(prompts)} prompts")
    
    # 4.calculate metrics
    all_metrics:List[Dict[str, Any]] = []
    correct = 0
    for example, output in tqdm(zip(input_examples, responses), total=len(input_examples), desc="Evaluating MMLU"):
        result = parse_mmlu_response(example, output)
        score = 0
        if result and result == example["answer"]:
            correct += 1
            score = 1
        all_metrics.append({
            "example": example,
            "generation": output,
            "score": score
        })
    logger.info(f"Pass {correct} generations.")
    
    # 5.serialize examples, generations, and scores
    with open(output_path, 'w', encoding='utf-8') as fout:
        for metrics in all_metrics:
            fout.write(json.dumps(metrics) + "\n")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/mmlu/test"
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="cs336_alignment/prompts_safety/mmlu_zero_shot.prompt"
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
        default="/mnt/cs336-a5-supplement-results/mmlu_zero_shot_metrics.jsonl"
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