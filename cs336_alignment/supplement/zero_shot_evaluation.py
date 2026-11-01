from decimal import Decimal
from typing import Any
import re

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    # parse generated language model outputs into the predicted answer letter.  
    # If a response cannot be parsed, return None
    """
    解析器识别答案的能力也会影响最终测得的准确率
    不可能写一个解析器 100% 还原所有可能的 LLM 输出。
    LLM 输出是自由文本，理论上可以写出无穷多种表达同一个答案的方式，任何基于规则的解析器都只能覆盖有限的模式。
    所以这类任务的正确目标不是"100% 还原"，而是尽量提高解析成功率，同时对解析不了的返回 None
    """
    if not model_output or not model_output.strip():
        return None

    # 去掉常见 Markdown 强调符号。
    text = model_output.replace("**", "").replace("`", "").strip()

    def normalize(value: str) -> str:
        return " ".join(value.strip().rstrip(".。").split()).casefold()

    def parse_candidate(candidate: str) -> str | None:
        candidate = candidate.strip()

        # 支持 B、(B)、[B]、B. 以及 B. explanation。
        # 不接受 a reliable answer、B or C 等表达。
        match = re.fullmatch(
            r"(?:([A-D])|\(([A-D])\)|\[([A-D])\])"
            r"(?:\s*[.:：。]\s*.*)?",
            candidate,
            flags=re.IGNORECASE,
        )
        if match:
            return next(g for g in match.groups() if g).upper()

        # 完整匹配选项原文；不使用子串匹配或模糊匹配。
        matches = [
            letter
            for letter, option in zip("ABCD", mmlu_example["options"])
            if normalize(option)
            and normalize(candidate) == normalize(option)
        ]
        return matches[0] if len(matches) == 1 else None

    # 只定位明确的答案声明，不直接从整段文字提取字母。
    declarations = list(re.finditer(
        r"\b(?:the\s+)?(?:(?P<kind>final|correct)\s+)?answer"
        r"\s*(?:is\b\s*:?\s*|[:：=]\s*)",
        text,
        flags=re.IGNORECASE,
    ))

    if not declarations:
        return parse_candidate(text)

    # 若有显式 final answer，只考虑这些最终答案声明。
    has_final = any(
        (match.group("kind") or "").lower() == "final"
        for match in declarations
    )

    predictions = []
    for i, match in enumerate(declarations):
        if has_final and (match.group("kind") or "").lower() != "final":
            continue

        end = declarations[i + 1].start() if i + 1 < len(declarations) else len(text)
        tail = text[match.end():end].strip()
        candidate = tail.splitlines()[0] if tail else ""
        predictions.append(parse_candidate(candidate))

    # 任何声明无法解析，或出现互相冲突的选择，都保守返回 None。
    if not predictions or None in predictions:
        return None

    return predictions[0] if len(set(predictions)) == 1 else None

def parse_gsm8k_response(model_output: str) -> str | None:
    if not model_output or not model_output.strip():
        return None

    # 排除模型续写的下一轮问答，仅解析当前回答。
    text = re.split(
        r"^\s*#\s*Query\s*:",
        model_output,
        maxsplit=1,
        flags=re.MULTILINE | re.IGNORECASE,
    )[0].replace("−", "-")

    # 支持整数、负数、小数、千位逗号及科学计数法。
    # 数字前后的货币符号、标点不会阻止匹配。
    numbers = re.findall(
        r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?",
        text,
    )
    if not numbers:
        return None

    # 统一数值表示，例如 $9.00 → "9"，1,200 → "1200"。
    result = format(Decimal(numbers[-1].replace(",", "")), "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")

    return "0" if Decimal(result) == 0 else result