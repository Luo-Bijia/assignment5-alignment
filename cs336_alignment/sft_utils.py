import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def tokenize_prompt_and_output(prompt_strs: list[str], 
                               output_strs: list[str], 
                               tokenizer: PreTrainedTokenizer
                               ) -> dict[str, torch.Tensor]:
    # Tokenize the prompt and output strings, and construct a mask aligned with labels that is 1 for response tokens and 0 for other tokens (prompt or padding).
    """
    The returned dictionary should have the following keys:

    ‣ input_ids : the tokenized prompt and output strings, with the final token sliced off.  
    ‣ labels : shifted input ids, i.e., the input ids without the first token.
    ‣ response_mask : a mask aligned with labels, with value 1 where the corresponding label token is part of the response and 0 otherwise.
    """
    batch_size = len(prompt_strs)
    prompt_ids_list = tokenizer(prompt_strs, add_special_tokens=False, padding=False)["input_ids"]
    resp_ids_list = tokenizer(output_strs,add_special_tokens=False, padding=False)["input_ids"]

    full_ids_list = [ids1+ids2 for ids1, ids2 in zip(prompt_ids_list, resp_ids_list)]
    full_lens = [len(ids) for ids in full_ids_list]
    max_len = max(full_lens)

    # prompt+response(output) 作为 RL 的 input，shifted prompt+response就是 RL 的label
    # 保证 prompt 和 response 各自 encode 之后直拼在一起（中间不插入pad），再右 padding 至 max_len
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
    unshifted_resp_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i in range(batch_size):
        full_ids = full_ids_list[i]
        full_len = full_lens[i]
        prompt_len = len(prompt_ids_list[i])

        padded_ids[i, :full_len] = torch.tensor(full_ids, dtype=torch.long)
        unshifted_resp_mask[i, prompt_len:full_len] = True      # prompt、pad 位置均为 False

    input_ids = padded_ids[:, :-1]
    labels = padded_ids[:, 1:]
    response_mask = unshifted_resp_mask[:, 1:]      # 指示 labels
    return {        # all shape is (batch_size, max(prompt_and_output_lens) - 1)
        "input_ids":input_ids,  
        "labels":labels,
        "response_mask":response_mask
    }

def get_response_log_probs(model: PreTrainedModel, 
                           input_ids: torch.Tensor, 
                           labels: torch.Tensor, 
                           return_token_entropy: bool = False
                           ) -> dict[str, torch.Tensor]:
    # gets per-token conditional logprobabilities (given the previous tokens) from a causal language model, and optionally the entropy of the model’s next-token distribution.
    outputs = model(input_ids)
    logits = outputs.logits
    all_log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V), 对 logits 先 softmax 再log - 是概率的对数形式，log p(x), 取值范围：(-∞, 0]
    index = labels.unsqueeze(dim=2)     # (B, T, 1)
    log_probs = torch.gather(input=all_log_probs, dim=-1, index=index).squeeze(2)       # (B, T) 沿词表维取出对 label token 计算的log prob
    result: dict[str, torch.Tensor] = {}
    result["log_probs"] = log_probs     # (batch_size, sequence_length)

    if return_token_entropy:
        all_probs = all_log_probs.exp()
        # entropy = -Σp*log(p)
        token_entropy = -torch.sum(all_log_probs * all_probs, dim=-1, keepdim=False)    # (B, T)
        result["token_entropy"] = token_entropy     # (batch_size, sequence_length)

    return result
