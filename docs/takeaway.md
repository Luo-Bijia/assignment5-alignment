先把每条 rollout 变成一个标量 advantage，再把这个 advantage 乘到该回答每个 token 的 log-prob 上（**这一步就是策略梯度最关键的“桥梁”：
它把离散、不可微的“这个回答应该被偏好还是压低”，转化成了可以沿模型计算图反向传播的连续 loss**）<—>最后对 token、回答和 microbatch 聚合，完成一次 optimizer update。


## 一、先固定几个维度

假设一次 rollout batch 中：

- `B`：问题数量
- `G`：每道题生成几个回答
- `N = B × G`：rollout 总数
- `T`：padding 后的 token 序列长度减一
- `V`：词表大小

例如：

- 2 道题
- 每题生成 3 个回答
- 那么共有 `N=6` 条训练样本

三个字符串列表必须这样对齐：

```text
repeated_prompts:
[x₀, x₀, x₀, x₁, x₁, x₁]

rollout_responses:
[y₀₀, y₀₁, y₀₂, y₁₀, y₁₁, y₁₂]

repeated_ground_truths:
[a₀, a₀, a₀, a₁, a₁, a₁]
```

`compute_group_normalized_rewards` 默认每连续 `G` 条属于同一道题，所以这个排列顺序非常重要。

---

# 二、完整数据流

```text
vLLM 生成字符串
      │
      ▼
rollout_responses
      │
      ├── compute_rollout_rewards
      │          ↓
      │     raw_rewards [N]
      │
      ├── compute_group_normalized_rewards
      │          ↓
      │      advantages [N]
      │
      └── tokenize_prompt_and_output
                 ↓
       input_ids / labels / response_mask [N,T]
                 │
                 ▼
          get_response_log_probs
                 ↓
        log_probs / token_entropy [N,T]
                 │
                 ▼
       compute_policy_gradient_loss
                 ↓
          per-token loss [N,T]
                 │
                 ▼
       aggregate_loss_across_microbatch
                 ↓
              scalar loss
                 │
                 ▼
           backward 累积梯度
                 │
                 ▼
      grad norm → clip → optimizer.step
```

vLLM 不在 `grpo_train_step` 里面。它在外层训练循环中提前生成 `rollout_responses`，然后把这些字符串交给 `grpo_train_step`。

---

# 三、`tokenize_prompt_and_output`

它解决的问题是：

> 如何把 prompt 和已经生成好的 response 变成 causal LM 可以重新评分的 token 张量，并标记哪些 label 属于 response。

## 输入

- `prompt_strs`：长度为 `N` 的 prompt 列表
- `output_strs`：长度为 `N` 的 response 列表
- `tokenizer`

一项列表元素代表一条完整样本，不是字符列表。

## 为什么分开 tokenize

你当前代码分别 tokenize：

- prompt
- response

然后拼接 token IDs。

这是作业明确要求的行为。它保证 prompt/response 的 token 边界由两个独立编码结果决定，不会因为直接拼字符串而在边界处产生新的 BPE 合并。

## 移位关系

假设一条样本 tokenize 后是：

```text
prompt tokens:   [p₁, p₂]
response tokens: [r₁, r₂]
完整序列:         [p₁, p₂, r₁, r₂]
```

产生：

```text
input_ids:     [p₁, p₂, r₁]
labels:        [p₂, r₁, r₂]
response_mask: [ 0,  1,  1]
```

对应关系是：

| 模型看到的前缀 | 要预测的 label | 是否属于 response |
|---|---|---:|
| `p₁` | `p₂` | 0 |
| `p₁,p₂` | `r₁` | 1 |
| `p₁,p₂,r₁` | `r₂` | 1 |

这里最关键的是：

> `response_mask` 必须和 `labels` 对齐，而不是和未移位的完整序列对齐。

所以你最后使用 `unshifted_resp_mask[:, 1:]` 是在跟随 labels 一起移位。

## 输出形状

```text
input_ids:     [N,T]
labels:        [N,T]
response_mask: [N,T]
```

padding 部分的 mask 是 `False`，prompt label 部分也是 `False`，只有 response label 是 `True`。

---

# 四、`get_response_log_probs`

这个函数解决的问题是：

> 模型对这条已经生成好的回答中的每个 token，到底赋予了多大的条件概率？

它不是让模型重新生成，而是给固定 response 打分。

## 模型输出

输入 `input_ids [N,T]` 后，causal LM 返回：

```text
logits: [N,T,V]
```

其中位置 `t` 的 logits 表示：

\[
p_\theta(\text{下一个 token}\mid \text{截至当前位置的前缀})
\]

由于 `labels` 已经提前左移，所以：

- `logits[:, t, :]`
- 正好对应 `labels[:, t]`

## `log_softmax`

你对词表维执行 `log_softmax`，得到：

```text
all_log_probs: [N,T,V]
```

每个位置都有整个词表的 log-prob 分布。

## `gather`

`labels` 给出了每个位置实际发生的 token ID。

`gather` 就是在词表维中取出那个实际 token 的 log-prob：

\[
\log \pi_\theta(y_t\mid x,y_{<t})
\]

结果是：

```text
log_probs: [N,T]
```

注意它目前包含：

- prompt token 的 log-prob
- response token 的 log-prob
- padding token 的 log-prob

这里只负责计算，还没有筛选。后面通过 `response_mask` 才只保留 response。

## token entropy

entropy 计算的是：

\[
H_t=-\sum_{v=1}^{V}p(v)\log p(v)
\]

它和 `log_probs` 的区别是：

- `log_probs[n,t]`：实际 label token 这一个 token 的 log-prob
- `token_entropy[n,t]`：该位置整个词表预测分布的不确定程度

直观上：

- entropy 高：模型在很多 token 之间犹豫
- entropy 低：模型非常确信某几个 token

你已经移除了之前的 `@torch.inference_mode()`，这是正确的，因为训练时 `log_probs` 必须保留计算图，才能从 policy-gradient loss 反向传播到模型参数。

---

# 五、`compute_rollout_rewards`

这个函数解决的问题是：

> 每条完整回答最终得了多少分？

## 输入

- `rollout_responses`：长度 `N`
- `repeated_ground_truths`：长度 `N`
- `reward_fn(response, ground_truth)`

每次 reward function 返回类似：

- `reward`
- `format_reward`
- `answer_reward`

## 输出

### `raw_rewards`

```text
raw_rewards: [N]
```

每条完整 response 只有一个标量 reward。

例如：

```text
[1, 0, 1, 0, 0, 0]
```

这里不是每个 token 一个 reward。GSM8K 的 reward 是在整个回答完成后统一计算的。

### metadata

你同时计算了整个 rollout batch 的：

- 平均 total reward
- 平均 format reward
- 平均 answer reward

这些应该用于训练日志。

---

# 六、`compute_group_normalized_rewards`

这个函数完成 GRPO 的“Group Relative”部分：

> 不直接看一条回答的绝对 reward，而是看它相对于同一道题的其他回答表现如何。

## 输入形状

```text
raw_rewards: [N]
```

每连续 `G` 项是一组。

假设一道题的四个回答 reward 是：

```text
[1, 0, 0, 1]
```

组均值是：

\[
\mu=0.5
\]

减去均值后：

```text
[0.5, -0.5, -0.5, 0.5]
```

再除以组内标准差，得到 normalized advantages。

## advantage 的意义

- `A > 0`：这条回答比同组平均更好，应提高概率
- `A < 0`：比同组平均更差，应降低概率
- `A = 0`：没有相对学习信号

如果同一道题的全部回答 reward 都一样：

```text
[0,0,0,0]
```

那么减去均值后全是 0，最终 advantage 也全是 0。

这一步不会让模型学习，因为这一组没有提供“哪个回答更好”的相对信息。

## 输出

```text
advantages: [N]
```

仍然是每条完整回答一个标量。

你使用 `advantage_eps + std` 防止标准差为零时除零。

---

# 七、`compute_policy_gradient_loss`

这个函数把：

- 每条 response 的标量 advantage
- 每个 token 的 log-prob

结合起来。

## 输入

```text
advantages:      [N]
policy_log_probs:[N,T]
```

你的 `unsqueeze(1)` 把 advantage 变成：

```text
[N,1]
```

然后沿 token 维广播：

```text
[N,1] × [N,T] → [N,T]
```

也就是说，同一条 response 的所有 token 共享相同 advantage。

这是当前序列级 reward 设置下的 credit assignment：

> 整条回答得分好，就提高构成这条回答的所有 response token；整条回答得分差，就降低它们。

## 为什么有负号

理论目标是最大化：

\[
A\log\pi_\theta(y_t\mid x,y_{<t})
\]

但 PyTorch optimizer 默认做梯度下降，因此返回：

\[
-A\log\pi_\theta(y_t\mid x,y_{<t})
\]

于是：

- `A > 0` 时，下降这个 loss 会提高对应 token 的 log-prob
- `A < 0` 时，会降低对应 token 的 log-prob

## 输出

```text
per_token_policy_gradient_loss: [N,T]
```

此时 prompt 和 padding 位置也有数值，因为这个函数尚未使用 mask。它们会在下一步被清除。

你当前只实现了：

```text
importance_reweighting_method = none
```

所以：

- `old_log_probs`
- `cliprange`
- `response_mask`

暂时不参与 policy loss 计算。

这符合 standard on-policy 部分的范围。

---

# 八、`aggregate_loss_across_microbatch`

这个函数解决的问题是：

> 如何把 `[microbatch_size,T]` 的 per-token loss 聚合成一个 scalar，供 `backward()` 使用？

## 第一步：计算 response 长度

对 mask 沿 token 维求和：

```text
responses_len: [microbatch_size,1]
```

这里得到每条 response 实际有多少个 token，不包括：

- prompt
- padding

## 第二步：sequence normalization

当 `loss_normalization="sequence"`：

\[
L_j=
\frac{1}{\operatorname{len}(y_j)}
\sum_t m_{j,t}L_{j,t}
\]

也就是每条 response：

1. 除以自己的 response 长度
2. mask 掉 prompt 和 padding
3. 对 token 求和

这样长回答不会仅仅因为 token 更多而天然贡献更大的梯度。

## 第三步：对回答平均

最后对 microbatch 中的 sequence loss 求均值：

\[
L=\frac{1}{M}\sum_{j=1}^{M}L_j
\]

输出：

```text
loss: scalar
```

这就是可以调用 `backward()` 的 loss。

一个需要牢记的不变量是：每条 response 至少应有一个 token。否则 `responses_len=0` 时会除零。

---

# 九、`grpo_train_step` 如何把它们串起来

你当前对它的注释是准确的：

> 一个 `grpo_train_step` 等于一个 rollout batch 上的一次 optimizer update。

gradient accumulation 并没有把它变成多次训练更新，只是把一次大 batch 的 forward/backward 拆小。

## 阶段 1：整个 rollout batch 统一算 reward

先执行：

```text
字符串 responses
→ raw_rewards [N]
→ advantages [N]
```

必须在切 microbatch 之前做 group normalization。

原因是一个 group 的 `G` 个回答可能被切到不同 microbatch。如果进入 microbatch 后才计算均值和标准差，GRPO 的 group 就被破坏了。

你当前是在切分前完成这一步，逻辑正确。

## 阶段 2：统一 tokenize

整个 rollout batch 被转换为：

```text
input_ids:     [N,T]
labels:        [N,T]
response_mask: [N,T]
```

然后才沿 batch 维切 microbatch。

## 阶段 3：对每个 microbatch forward

每个 microbatch 中：

```text
input_ids_microbatch
      │
      ▼
model forward
      │
      ├── log_probs [M,T]
      └── entropy   [M,T]
```

这里的 `M` 是当前 microbatch 的样本数。

## 阶段 4：生成 per-token policy loss

```text
advantages_microbatch [M]
             +
log_probs_microbatch [M,T]
             ↓
per-token loss [M,T]
```

然后用对应的：

```text
response_mask_microbatch [M,T]
```

过滤 prompt 和 padding。

## 阶段 5：聚合成 scalar

每个 microbatch 得到一个 scalar loss。

但这个 scalar 是“当前 microbatch 内的平均”，不能直接把多个 microbatch loss 原样相加，否则不同大小的 microbatch 权重会不正确。

所以你乘上：

\[
\frac{\text{当前 microbatch 大小}}{\text{整个 rollout batch 大小}}
\]

然后再调用 `backward()`。

所有 microbatch 的梯度会累积在同一组 `parameter.grad` 中，最终效果相当于对完整 batch 求平均后 backward。

## 阶段 6：只更新一次参数

循环结束后：

1. 计算当前累积梯度的 global norm
2. 如果设置了 `max_grad_norm`，进行 gradient clipping
3. 调用一次 `optimizer.step()`
4. 调用一次 `optimizer.zero_grad()`

因此：

```text
多个 forward
+ 多个 backward
+ 一次 optimizer.step
= 一次 GRPO train step
```


---

# 十二、用一句话记住每个函数

| 函数 | 一句话作用 |
|---|---|
| `tokenize_prompt_and_output` | 把字符串变成错位一格的 LM 输入和 label，并标记 response token |
| `get_response_log_probs` | 查询 policy 给实际 label token 分配的条件 log-prob |
| `compute_rollout_rewards` | 给每条完整回答打一个最终分数 |
| `compute_group_normalized_rewards` | 把绝对 reward 变成同题回答之间的相对 advantage |
| `compute_policy_gradient_loss` | 用 advantage 决定每个 response token 的概率该升还是该降 |
| `aggregate_loss_across_microbatch` | mask 掉无关 token，并把 token loss 聚合成 scalar |
| `grpo_train_step` | 对一个 rollout batch 累积梯度并执行一次参数更新 |

最核心的整条链条就是：

\[
\boxed{
\text{回答字符串}
\rightarrow
\text{reward}
\rightarrow
\text{advantage}
\rightarrow
\text{token log-prob}
\rightarrow
-A\log p
\rightarrow
\text{mask/平均}
\rightarrow
\text{backward/update}
}
\]
