# FinAgent-R1 面试问答手册

> 使用方式：先遮住答案，用自己的话回答 60–90 秒，再看要点。
>
> **诚实原则**：这个项目目前是"代码与实验协议完成、真实训练结果待跑"。如果面试官问效果，正确回答是"我还没有在 4×A40 上跑完，以下是我的评测协议和预期"，而不是编一个数字。下面很多题都给了"没有结果时怎么答"的模板。

---

## A. 项目与动机

### Q1. 用一句话介绍你的项目。

**答**：我在 Agent-R1 的 step-level MDP 框架上做了一个金融 filings 的 agentic RL 项目：用 Qwen2.5-3B 作为策略，配备一个按公司/财年 scoped 的 BM25 检索器和一个 AST 安全计算器，在 FinQA 的 8,281 道财报数值推理题上，用 GRPO/RLOO/REINFORCE/PPO 等多种算法做后训练，目标硬件是 4×A40 48GB。

**追问：一句话里最重要的词是什么？**
"step-level"——因为项目的核心不是"用 RL 做 QA"，而是把"检索→验证→计算→回答"的每一步当作 MDP transition 来优化。

### Q2. 为什么选金融领域？

**答**：三个原因：
1. **任务天然多步**：金融问答不是"背知识点"，而是"找报表行→核对口径→计算→给结论"，和 agentic RL 的 step-level 结构匹配；
2. **可验证**：FinQA 有 `exe_ans` 和 reasoning program，奖励可以规则化，不需要训练 reward model；
3. **失败模式清晰**：检索错、算错、格式崩，三类错误可以分别度量，便于做消融和 error analysis。

**追问：金融领域有什么特殊风险？**
数字不能错、口径要一致（GAAP vs non-GAAP、fiscal year vs calendar year）、数据有时效性和合规要求。项目里通过"只允许使用工具返回的数字"和"scope 限定同一份 filing"来降低风险；对外产品还需要免责声明和人工复核。

### Q3. 为什么不用 RAG / SFT / prompt ReAct？

**答**：
- RAG 是单轮检索，检索错了没有第二次机会；
- prompt ReAct 能多轮，但策略是"借来的"，7B 以下模型容易格式崩溃、重复查询；
- SFT 只能模仿专家轨迹，遇到第 1 步偏差不会自我纠偏；
- agentic RL 让模型通过多条轨迹的成败对比，自己学出"先检索、再计算、最后回答"的策略。

**追问：SFT 真的不行吗？**
不是。SFT 是很好的冷启动，工业上常见 "SFT + RL"。本项目先做 RL-from-base，是为了隔离变量；下一步会加 SFT 冷启动对照。

### Q4. 你的项目和 FinQA 原论文有什么区别？

**答**：FinQA 原论文主要是**单轮**的 program generation / numerical reasoning（给定 context，生成 program 或答案）。本项目把 context 拿掉，改成**多轮 agent**：模型必须自己调用检索器找证据、调用计算器算数，RL 优化的是整条交互轨迹。数据、答案、program 来自 FinQA，但任务形式和优化目标不同。

**追问：那你还算复现 FinQA 吗？**
不算。我不声称复现 FinQA 的 leaderboard；我把它当作一个可验证的 agentic RL testbed。评测时也区分 seen/unseen 文档，避免和原论文的数字直接对比。

### Q5. 项目最大的创新点是什么？

**答**：分三层说：
1. **工程隔离**：`core.py` 不依赖 verl/agent_r1，训练、vLLM 离线评测、单测共用同一套 prompt/tool/parser/reward，避免"训练一套、评测一套"；
2. **上下文重塑**：evidence scratchpad + sliding observation window + 分级 prompt budget，环境而不是模型决定下一步看到什么；
3. **可审计的数据协议**：官方 FinQA split + `doc_seen_in_train` 标记 + 数据集卡，明确区分"见过文档"和"真正泛化"。

**追问：这些算学术创新吗？**
不算颠覆性创新，更像"把 step-level MDP 落地到金融多跳任务"的系统性工程和实证研究。学术上可做的是 step-level reward shaping、retrieval-aware credit assignment，这些我列为后续工作。

### Q6. 项目目前的状态？效果如何？

**答（诚实版）**：代码、数据管线、5 种算法脚本、CPU 测试（60 个）和评测脚本都完成了；真实训练还没在 4×A40 上跑完，所以我没有可以负责的准确率数字。我的评测协议是：dev 选 checkpoint，test 只评一次，同时报告 all / seen_docs / unseen_docs，并用 paired bootstrap 给置信区间。

**追问：那你凭什么说它 work？**
目前能说的是"链路可运行"：CPU 单测覆盖了解析、奖励、BM25、AST 沙箱、完整 episode loop、prompt 预算和数据转换；端到端 smoke test 能生成数据并用 gold answer 得到 1.1 的 reward。模型效果必须等实机结果。

---

## B. 数据

### Q7. 数据规模和划分？

**答**：FinQA 共 8,281 道题，来自 2,789 份 S&P 500 财报。官方 split：train 6,251 / dev 883 / test 1,147，约 75.5% / 10.7% / 13.9%。我保持官方 split 不变，以便和文献对比。

### Q8. 训练/验证/测试怎么用？

**答**：
- train 只训练；
- dev 每 25 个 update 验证一次，用于选 checkpoint / 早停；
- test 只在最终评一次；
- 另外按 `doc_id` 标记 test 中"文档是否在 train 出现"，分别报告 seen/unseen。

### Q9. 官方 split 有什么问题？

**答**：它是 **question-level** 而不是 **document-level**。同一份 10-K 可能同时有训练题和测试题，模型可能记住这份报表的格式/数字，导致 test 分数虚高。所以我加了 `doc_seen_in_train`，并强制报告 `unseen_docs` 子集。

**追问：为什么不直接按文档重新划分？**
会失去和 FinQA 文献的可比性，而且文档级划分后 test 分布会变（某些公司完全没在 train 出现）。我的做法是"主表用官方 split，附表用 unseen 子集"，两者都报。

### Q10. 检索语料怎么构建？

**答**：每份 filing 拆三类 chunk：
- `pre_text`：表格前正文，按句切，≤700 字符；
- `table`：第一行做表头，每 10 行一个 chunk，**每个 chunk 重复表头**；
- `post_text`：表格后正文。
每个 chunk 带 `chunk_id/doc_id/ticker/year/kind/text`。2,789 份文档预计 3–7 万 chunk。

**追问：为什么表头要重复？**
表格 chunk 被单独检索后，模型看不到原表头就不知道列是 2021 还是 2020。重复表头是用少量冗余换取可读性和正确性。

### Q11. 表格怎么序列化？为什么不用 HTML/Markdown 表格？

**答**：用 pipe-separated 行：`|  | 2021 | 2020 |` + `| Revenue | 100.0 | 80.0 |`。原因：
- token 效率比 HTML 高；
- 行列关系对 LLM 可读；
- 比 Markdown 对齐表更抗换行/空格噪声；
- 方便 BM25 对数字和行名做匹配。
HTML 保留更多结构，但 token 多；复杂合并单元格 FinQA 里较少，pipe 足够。

### Q12. 如何防止答案泄漏？

**答**：
- 不把 `program`、`exe_ans`、`gold_evidence` 放进 prompt；
- 训练数据里这些字段只在 `extra_info`，reward 函数读取，模型看不到；
- 检索语料只含正文和表格，不含 QA 对；
- `doc_seen_in_train` 用于评测时区分记忆和泛化。

**追问：corpus 里会不会有答案？**
可能。财报正文本身可能直接写"revenue grew 25%"，这是合法证据，不是标注泄漏。我们禁止的是把 FinQA 的 `answer` 字段写进语料。

### Q13. 奖励怎么从 FinQA 标注得到？

**答**：优先用 `exe_ans`（执行 program 的结果）做数值匹配；没有 `exe_ans` 时回退到 `answer` 字符串/ token F1。匹配容差：相对 1e-3、绝对 1e-4；百分比和小数等价（25% ↔ 0.25）。另外加格式奖励（analysis/answer 各 0.05）、步数效率折扣、工具错误扣分。

### Q14. 数据质量怎么保证？

**答**：`dataset_stats.py` 输出数据卡：
- split 行数、唯一文档/ticker；
- 数值题比例；
- question 长度 p50/p95；
- program 长度直方图；
- 文档重叠行数；
- corpus chunk 数/每文档分布/scope 覆盖。
人工抽 20 题核对 question/answer/scope；如果 test 行在 corpus 中找不到同 ticker/year 的 chunk，就修数据，不训练。

### Q15. 为什么选 FinQA，不选 TAT-QA / FinanceBench / ConvFinQA？

**答**：
- **TAT-QA**：16,552 题，规模更大，但答案类型混合（span/arithmetic/count），奖励更复杂，表格更宽；适合作为第二数据集，不是第一版。
- **FinanceBench**：更接近真实 open-book 10-K QA，但需要 PDF 解析、证据页定位，工程量大；10,231 题里很多是文本答案，数值 program 少。
- **ConvFinQA**：多轮对话，适合研究对话状态，但和 step-level 工具调用不是同一类任务。
- **FinQA**：8,281 题、有 program/exe_ans、表格+正文、规模适中，最适合在 4×A40 上做第一版。

**追问：只用 FinQA 会不会过拟合一个数据集？**
会。下一步会加 TAT-QA 或 FinanceBench 的数值子集做 cross-dataset 测试；当前先保证单数据集链路正确。

---

## C. Agentic RL 与 MDP

### Q16. 什么是 step-level MDP？

**答**：把一次 LLM agent 交互建模为 $(\mathcal S,\mathcal A,\mathcal T,\mathcal R,\gamma,H)$，其中"一步"= 一次 LLM 调用。$s_t$ 是环境构造的 prompt（system + 问题 + scratchpad + 最近观测），$a_t$ 是模型输出（analysis + tool calls 或 answer），$\mathcal T$ 执行工具并构造 $s_{t+1}$，$r_t$ 是即时奖励，$H$ 是最大步数。

### Q17. 和 token-level MDP 的区别？

**答**：
- token-level 把每个 token 当时间步，工具观测和 action token 混在一条序列里；
- step-level 把一次 LLM 调用当时间步，工具观测是独立字段、`response_mask=0`；
- step-level 允许环境在步骤之间裁剪/重写上下文；
- 信用分配可以在 step 级（GAE/组比较），不必落到每个 token。

### Q18. 你的 state/action/reward 具体是什么？

**答**：
- state：system prompt + 问题 + `[Filing scope: TICKER, year]` + evidence scratchpad + 最近 2 条 observation（每条 ≤3500 字符）+ 工具错误/格式反馈 + step 计数；
- action：`<analysis>...</analysis>` + 1–2 个 `<tool_call>{"name":...,"arguments":...}</tool_call>`，或 `<answer>...</answer>`；
- reward：中间步 0；终止步 = outcome（numeric match 或 F1）× efficiency + format(0.1) − tool penalty(≤0.15)。

### Q19. 为什么让环境控制上下文，而不是模型自己 append？

**答**：
- 模型没有"删除历史"的动作，append-only 必然导致上下文爆炸；
- 环境知道哪些观测是验证过的（计算结果）、哪些是冗余的（重复检索）；
- 环境可以强制保留 system prompt 和最近证据，丢掉中间噪声；
- 这也是 Agent-R1 的设计理念：state 是环境的责任，不是 token 序列的副作用。

### Q20. 多轮信用分配难在哪？

**答**：终止奖励只知道"整条轨迹好不好"，不知道"第 2 步的检索错了还是第 3 步的计算错了"。可选方案：
1. 轨迹级组比较（GRPO/RLOO）：把整条轨迹当一个动作；
2. step-level critic/GAE：学 $V(s_t)$；
3. step-level reward shaping：检索命中/计算正确给部分奖励；
4. GiGPO：按 anchor observation 分组做 step 级相对优势。
本项目默认 1，提供 2 的两种粒度，3/4 是后续。

### Q21. 奖励稀疏怎么办？

**答**：
- 格式奖励（0.1）先让模型学会输出协议；
- efficiency 折扣鼓励少走弯路；
- tool penalty 抑制非法调用；
- 课程学习：先用较短 `max_steps` 或简单题，再放开；
- 后续可加 step-level shaping，但要防止 hack。
如果模型从不输出 `<answer>`，先跑 base model 看格式率；必要时用 SFT 冷启动。

### Q22. 停止条件是什么？

**答**：
- 模型输出非空 `<answer>` → 停止；
- 模型没有 tool call 也没有 answer（纯 prose）→ 停止（视为失败）；
- malformed tool call 且不是最后一步 → 不停止，给反馈让它重试；
- 到 `max_steps=6` → 强制停止。
终止步的响应参与 reward；如果最后一步只有 tool call 没有 answer，outcome=0。

### Q23. 为什么 malformed call 要消耗一步而不是直接终止？

**答**：直接终止会让模型失去"从格式错误中恢复"的学习信号。给一次纠正提示、消耗一步，能让策略学会"看到反馈→修正 JSON"，这也是一种自我纠错能力。代价是可能浪费步数，所以 efficiency 折扣会惩罚过多无效步。

### Q24. 多轮轨迹怎么 batch？为什么不能按行 shuffle？

**答**：同一轨迹的多个 step 必须一起做 advantage/GAE 递归。Agent-R1 用 `trajectory_uids` + `step_indices` 标识，`trajectory_batching.py` 按轨迹构造 mini-batch，并用 `sample_mask` 标记 padding。如果按行 shuffle，同一条轨迹的 step 会被拆到不同卡，GAE 的跨 step 递推和 GRPO 的轨迹回报都会错。

---

## D. 强化学习算法

### Q25. GRPO 的公式和直觉？

**答**：对每个 prompt 采样 $G$ 条轨迹，回报 $R_i$：

$$A_i=\frac{R_i-\operatorname{mean}(R)}{\operatorname{std}(R)+\epsilon}$$

同一轨迹的所有 step/token 共享 $A_i$。直觉：不看绝对分数，只看"这条轨迹比同组平均好多少"，因此不需要 critic，也自然消除了题目难度差异。

### Q26. GRPO 为什么不需要 critic？

**答**：critic 的作用是估计 baseline 降方差。GRPO 用"同 prompt 的多条采样"构造 baseline（组均值），用组标准差归一化。代价是需要 $G$ 条 rollout，rollout 成本高；好处是省掉一个和 actor 同样大的 value 网络，显存和工程复杂度都低。

### Q27. GRPO 的 $G$ 怎么选？$G=1$ 会怎样？

**答**：$G$ 越大，均值和方差估计越稳，但 rollout 成本线性增长。常用 8；$G=2$ 也能跑但方差大。$G=1$ 时没有组内比较，代码会把 std 设为 1、均值设为 0，advantage 退化成 $R_i$（且没有 baseline），基本等于 REINFORCE，不建议。

### Q28. RLOO 和 GRPO 的区别？

**答**：RLOO 用留一均值做 baseline：

$$A_i=R_i-\frac{1}{G-1}\sum_{j\ne i}R_j$$

不做 std 归一化。GRPO 用组均值 + 组标准差。RLOO 的 baseline 是无偏的，通常方差略低；GRPO 的 std 归一化会让不同难度组的梯度尺度更一致，但也可能放大简单题的噪声。

### Q29. REINFORCE 在你们框架里怎么实现？

**答**：step-level 折扣回报 $G_t=\sum_{k=t}^{H}\gamma^{k-t}r_k$，然后对整个 batch 的 step return 做白化 $A_t=(G_t-\mu)/\sigma$。没有 critic。实现见 `core_algos.compute_reinforce_outcome_advantage`。

**追问：为什么白化是 batch 级而不是组级？**
这是上游实现的选择。batch 级白化简单，但会把不同 prompt 的难度混在一起；组级白化（GRPO）通常更稳。可以作为一个改进点。

### Q30. PPO / GAE 的公式？

**答**：

$$\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$$
$$A_t=\sum_{l=0}^{H-t}(\gamma\lambda)^l\delta_{t+l}$$

PPO 目标：

$$L=\mathbb E[\min(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]$$

其中 $\rho_t=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$。Agent-R1 的 `gae` 是 step-level：每个 step 一个 $V$，GAE 跨 step 递推。

### Q31. StepPO 是什么？和 PPO 有什么区别？

**答**：StepPO 的核心思想是**以 step 为对齐/信用分配单位**，而不是 token。Agent-R1 并没有一个叫 `steppo` 的 estimator；上游 `examples/gsm8k/run_steppo.sh` 用的是 `algorithm.adv_estimator=gae`，配合 step-level 的轨迹和奖励。所以本项目的 `train_ppo.sh` 既是 PPO，也是 Agent-R1 的 StepPO-style 复现。

**追问：那它和普通 PPO 到底差在哪？**
差在 MDP 的时间单位和 mask：普通 PPO 把整条 token 序列当一个 episode；这里每个 step 是一个 transition，工具观测被 mask 掉，GAE 跨 step 递推，critic 只在 step 起点估 value。

### Q32. Token-GAE 和 Step-GAE 的区别？

**答**：
- Step-GAE：critic 每个 step 一个 value；GAE 在 step 之间递推；value loss 只取每个 step 的第一个 action token。
- Token-GAE：critic 每个 action token 一个 value；GAE 在 token 之间递推，跳过 `response_mask=0` 的工具观测，不推进递归。
Token-GAE 信用分配更细，但 critic 噪声/显存更大；多轮工具场景通常 Step-GAE 更稳。

### Q33. GiGPO 是什么？为什么没用？

**答**：GiGPO 在 GRPO 的轨迹级相对优势之外，再按 `anchor_obs` 把不同轨迹中"相同/相似 observation"的 step 分到一组，算 step-level 相对优势，最后加权。它适合有重复状态的環境（如 ALFWorld 的房间）。FinQA 的 observation 是检索文本，几乎不重复，anchor 分组收益有限；而且需要 flow 提供 `anchor_obs`，所以本项目未启用。

### Q34. KL 散度起什么作用？

**答**：约束策略不要偏离初始 SFT/参考模型太远，防止 reward hacking 和语言能力退化。Agent-R1/verl 支持：
- `use_kl_in_reward=True`：KL 加进 reward；
- `actor.use_kl_loss=True`：KL 作为 actor loss 的正则项（本项目用这个）；
- `kl_loss_type=low_var_kl`：低方差 KL 估计，数值更稳。
系数 0.001 是常用起点；太大模型学不动，太小会漂移。

### Q35. 熵正则的作用？

**答**：鼓励探索，防止策略过早塌缩到单一行为。系数 0.001 很轻。如果 entropy 迅速归零，说明探索不足，可以调大；如果 entropy 一直很高、reward 不升，说明太随机，可以调小。

### Q36. 优势归一化/白化有什么坑？

**答**：
- GRPO 的 std 归一化会放大低方差组的梯度；全对/全错的组 std=0，advantage=0，没有梯度；
- batch 级白化（REINFORCE）会把不同 prompt 混在一起；
- 白化后 advantage 的尺度不再反映真实回报，调试时不要用它当 reward 看。
可以试 `norm_adv_by_std_in_grpo=False`（Dr.GRPO）做消融。

### Q37. 如果所有 rollout 都成功/都失败怎么办？

**答**：GRPO/RLOO 的组内 advantage 全为 0，这个 prompt 不贡献梯度。解决：
- 课程学习/难度采样，避免全对全错；
- 保留部分简单题维持信号；
- 加 step-level reward shaping，让组内有差异；
- 动态采样（只训练有区分度的组），但会增加实现复杂度。

### Q38. 你会推荐哪个算法作为默认？为什么？

**答**：先 GRPO。原因：
- 不需要 critic，4×A40 显存压力小；
- 实现成熟，组比较直观；
- 在 agent 任务上通常是一个强 baseline。
然后跑 RLOO 做 baseline 对比；PPO 用于验证 critic 是否值得；Token-GAE 作为信用分配消融。最终选哪个要看 dev 准确率/稳定性/成本，而不是先验。

---

## E. Agent-R1 框架

### Q39. Agent-R1 的三层抽象是什么？

**答**：
1. `AgentFlowBase`：完全自定义流程（prompt、分支、上下文、step 打包）；
2. `AgentEnvLoop` + `AgentEnv`：标准 `reset/step` 环境循环；
3. `ToolEnv` + `BaseTool`：只定义工具，环境自动解析和调用。
本项目用 Layer 1，因为需要 scratchpad、滑动窗口、动态 prompt 预算和 scoped 工具。

### Q40. `AgentFlowBase.run` 的契约是什么？

**答**：
- 接收 `sampling_params` 和 dataset row 的 `**kwargs`；
- 每轮生成后构造 `AgentFlowStep(prompt_ids, response_ids, reward_score, ...)`；
- 必须 `await self._postprocess(step, **kwargs)`；
- 中间步 `reward_score=0.0`，终止步 `reward_score=None`（触发 reward worker）；
- 返回 `AgentFlowOutput(steps, metrics)`。

### Q41. `AgentFlowStep` 里有什么？

**答**：`prompt_ids`、`response_ids`、`input_ids`、`position_ids`、`attention_mask`、`response_mask`、`response_logprobs`、`routed_experts`、`multi_modal_data`、`reward_score`、`num_turns`、`extra_fields`。核心是 `response_mask`：1=LLM token，0=工具观测/padding。

### Q42. `response_mask` 有什么用？

**答**：
1. policy loss 只在 LLM token 上算，工具观测不回传梯度；
2. advantage 广播时乘 mask，避免观测 token 被更新；
3. reward 定位到最后一个有效 action token；
4. critic 的 value mask 也基于它。

### Q43. `trajectory_uids` 和 `step_indices` 有什么用？

**答**：把展开后的 step 行还原成轨迹：
- `trajectory_uids`：同一条轨迹的所有 step 共享一个 uid；
- `step_indices`：step 在轨迹内的顺序 0..T-1；
- GRPO/RLOO 用 uid 汇总轨迹回报；
- GAE 用 uid+step_index 做跨 step 递推；
- trajectory batching 用它们保证同轨迹不被拆开。

### Q44. verl / Ray / vLLM / FSDP 各自负责什么？

**答**：
- **verl**：提供 PPO 训练栈、DataProto、worker 抽象、FSDP/vLLM 集成、checkpoint；
- **Ray**：分布式 actor 调度、资源池、把 driver/worker 串起来；
- **vLLM**：rollout 推理，PagedAttention + continuous batching；
- **FSDP**：训练时的参数/梯度/优化器分片（ZeRO-3），前后向 all-gather/reduce-scatter。
Agent-R1 在 verl 之上加了 step-level 轨迹、AgentFlow、reward loop 和 advantage 扩展。

### Q45. 一次训练的数据流？

**答**：parquet → `RLHFDataset` → `AgentFlowWorker.generate_sequences` → `FinQAAgentFlow.run` → `core.run_episode`（vLLM 生成 + 工具执行）→ 每个 turn 包成 `AgentFlowStep` → `_postprocess`（padding/mask/reward）→ `DataProto` → `RayAgentTrainer.fit`（repeat n、sample-level repeat、union、KL、advantage、critic/actor update）。

### Q46. 怎么注册一个新的 agent flow？

**答**：
1. 继承 `AgentFlowBase`，实现 `run`；
2. `@register("my_agent")`；
3. YAML 里写 `- name: my_agent` + `_target_: my.module.MyFlow`；
4. 训练时传 `agent_flow_config_path` 和 `default_agent_flow=my_agent`。
也可以不用装饰器，只要 YAML 的 `_target_` 正确，registry 会在加载 YAML 时填充。

### Q47. 怎么加一个新工具？

**答**：
1. 继承 `BaseTool`，设置 `name/description/parameters`；
2. 实现 `async execute(args, **kwargs) -> (ToolResponse, reward, meta)`；
3. `@BaseTool.register("my_tool")`；
4. 如果走 `ToolEnv`，它会自动从 registry 调用；如果走自定义 flow，在 flow 里实例化并调用。
注意 `parameters` 必须是 OpenAI function schema 的 properties。

---

## F. 系统与工程

### Q48. 4×A40 48GB 的显存账本？

**答（3B，bf16，FSDP ZeRO-3）**：
- 参数 6.2GB + 梯度 6.2GB + AdamW(fp32 m+v) 24.8GB + Reference 6.2GB = 43.4GB；
- 4 卡均摊 ≈ 10.9GB/卡；
- vLLM 0.45×48 ≈ 21.6GB；
- 激活（micro-batch 4 + 梯度检查点）约 5–8GB；
- 峰值约 38–41GB/卡。
PPO 再加一个 3B critic，约 +10.9GB/卡，所以 vLLM 要降到 0.35。

### Q49. 为什么选 3B 而不是 7B？

**答**：7B 在 4×48GB 上：
- 静态占用 106.4/4 ≈ 26.6GB/卡；
- 加 vLLM 0.30–0.35 和激活，约 49–56GB，已经 OOM；
- 必须 param/optimizer offload 或 LoRA，吞吐明显下降；
- PPO 还要再加一个 7B critic，基本不可行。
3B 可以无 offload 跑 GRPO/RLOO/REINFORCE，迭代快，适合个人项目。7B 作为可选配置，不默认。

### Q50. 为什么 PPO 显存更高？

**答**：PPO 需要 critic（value network），参数量与 actor 同级：参数+梯度+优化器状态再加一份。step-level GAE 还要存 values。显存和计算都接近翻倍，所以 PPO 脚本把 rollout n 从 8 降到 4、vLLM 比例降到 0.35。

### Q51. `gpu_memory_utilization` 怎么调？

**答**：它是 vLLM 可用的显存比例（含模型权重和 KV cache）。
- 太小：KV cache 不足，rollout 排队/变慢；
- 太大：训练阶段 OOM（尤其在 colocate 模式）。
起点：GRPO 0.45，PPO 0.35。OOM 时按 0.40→0.35→0.30 降；rollout 慢时在确认训练有余量后升。

### Q52. OOM 排查顺序？

**答**：
1. `nvidia-smi` 看是 rollout 还是 train 阶段；
2. 降 `MICRO_BATCH_SIZE`（2）；
3. 降 `GPU_MEMORY_UTILIZATION`；
4. 降 `TRAIN_BATCH_SIZE` / `MINI_BATCH_SIZE` / `ROLLOUT_N`；
5. 确认梯度检查点开启；
6. PPO 开 critic param offload；
7. 最后才开 actor optimizer offload（最慢）。

### Q53. 4 卡 A40 无 NVLink，对训练有什么影响？

**答**：FSDP 的 all-gather/reduce-scatter 走 PCIe 4.0 x16（约 32GB/s 双向），比 NVLink 慢很多。因此：
- TP 尽量为 1（3B 单卡放得下），用 DP=4；
- 减少不必要的 all-gather（ZeRO-3 本身通信多）；
- micro-batch 不要太小，避免通信占比过高；
- 如果做 7B/TP>1，通信会成为瓶颈。

### Q54. 断点续训和 checkpoint 怎么管？

**答**：verl 保存到 `trainer.default_local_dir/global_step_N/actor`，并在 `latest_checkpointed_iteration.txt` 记录最新步。重新跑同一脚本会从最新 checkpoint 恢复。每个 3B 全量 checkpoint 约 25–35GB，`SAVE_FREQ=50` 也会占用几百 GB，需要定期清理或只保留最好的 1–2 个。

### Q55. 怎么把 FSDP checkpoint 变成 vLLM 可加载的 HF 目录？

**答**：用 verl 的 model merger：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir outputs/<exp>/global_step_N/actor \
  --target_dir outputs/<exp>/global_step_N/huggingface
```

然后 `MODEL=.../huggingface` 跑评测。注意 verl 版本不同，参数可能略有差异，以 `--help` 为准。

### Q56. 训练吞吐瓶颈通常在哪？

**答**：多轮 agent RL 通常是 **rollout 主导**：
- 每步都要等 vLLM 生成；
- 工具调用有 I/O/检索延迟；
- 长 prompt 的 prefill 很贵；
- 训练阶段 FSDP 通信在 PCIe 上也不便宜。
优化：vLLM continuous batching、减少步数、压缩 observation、BM25 缓存、并行工具调用、避免重复检索。

---

## G. 评估与实验设计

### Q57. 你的评估指标？

**答**：
- Numeric accuracy：`<answer>` 中数字与 `exe_ans` 匹配（rel 1e-3/abs 1e-4，百分比/小数等价）；
- Token F1：非数值答案；
- Composite reward：与训练一致；
- Answer rate / format rate；
- Avg steps / tool calls；
- Invalid-call rate / tool-error rate；
- Retrieval chunk 数；
- Retrieval recall：检索到的 observation 是否覆盖 FinQA gold evidence（前 80 字符归一化匹配，诊断用，不进 reward）。
另外按 all / seen_docs / unseen_docs 分组。

### Q58. 为什么要报告 seen/unseen？

**答**：官方 split 是问题级，同一份财报可能同时在 train/test。只报 all 会把"记住这份报表"误当成泛化。unseen_docs 子集要求模型在完全没见过的文档上检索和计算，更接近真实部署。

### Q59. 如何保证算法对比公平？

**答**：
- 同一模型、同一数据、同一 prompt/tool/reward；
- 同一 `max_steps`、`max_prompt_length`、`max_response_length`；
- 同一评测 temperature=0；
- 同一 checkpoint 选择规则（dev accuracy）；
- 同一 test 只评一次；
- 报告均值 ± 95% CI，最好 2 个 seed。
不同算法的最优超参可能不同，所以要么固定预算，要么报告调参预算。

### Q60. 怎么做统计显著性？

**答**：1,147 题上比较两个配置，用 paired bootstrap：对题目有放回采样 1,000 次，算准确率差的分布，取 2.5%/97.5% 分位。也可以用 McNemar 检验（配对二分类）。不要只看 0.5 个百分点就宣称更好。

### Q61. 消融怎么做？

**答**：
- 上下文：scratchpad on/off、sliding window 1/2/4、full context；
- 检索：BM25 vs dense、top_k 1/3/5、scope fallback on/off；
- 奖励：去 efficiency、去 tool penalty、去 format；
- 算法：GRPO/RLOO/REINFORCE/PPO/Token-GAE；
- 模型：3B vs 7B（如果显存允许）。
每次只改一个变量，固定 seed 和预算。

### Q62. 如果 base model 的 zero-shot 已经很高，RL 还有意义吗？

**答**：要看错误类型。如果 base 已经 80%+，RL 的增益可能集中在：
- 格式稳定性（非法 JSON、缺 answer）；
- 多步计算（心算错误）；
- 检索失败后的恢复；
- 步数效率。
如果 base 已经很好且没有这些错误，RL 的边际收益低，应该先做 error analysis 再决定。

### Q63. 怎么分析失败案例？

**答**：从 predictions 里抽样，按以下分类：
1. 检索错（错行/错年份/错公司）；
2. 计算错（公式错/算术错/百分比方向错）；
3. 格式错（无 answer/非法 JSON）；
4. 过早停止（明明有证据却回答）；
5. 死循环（重复同一 query）；
6. ground truth 歧义/数据错误。
每类给一个改进假设，并设计一个小实验验证。

---

## H. 失败、反思与行为

### Q64. 这个项目最大的技术挑战是什么？

**答**：不是模型，而是**让训练和评测使用完全相同的交互协议**。多轮 agent 的 prompt、工具返回、停止条件、reward 很容易在训练和评测之间漂移。我的解决方式是把 episode loop 抽成不依赖框架的 `core.py`，训练和 vLLM 评测都注入同一个 loop；单测直接测这个 loop。第二个挑战是显存：3B + vLLM + ref + critic 在 4×48GB 上要精打细算。

### Q65. 如果重做，你会改什么？

**答**：
1. 先做 SFT 冷启动，再 RL，减少早期格式探索成本；
2. 更早接稠密检索（BGE/FAISS），BM25 的同义词召回有限；
3. 加 step-level reward shaping（检索命中 gold evidence、计算表达式正确），但要先设计防 hack；
4. 用文档级 split 作为主实验，官方 split 作为附录；
5. 从第一天就记录 seed、显存、吞吐、失败案例，而不是最后补。

### Q66. 你从项目里学到的最重要的一件事？

**答**：**RL 的效果上限由环境和奖励决定，不是由算法名字决定。** 如果检索器找不到证据、奖励只看最终答案、上下文无限增长，换 GRPO/PPO/RLOO 都不会有质变。先把环境、数据、奖励做对，再谈算法。

### Q67. 怎么向非技术面试官解释这个项目？

**答**：把它比作"训练一个金融分析实习生"：
- 给他一个财报库（FinQA）；
- 给他两个工具（查资料、计算器）；
- 让他自己尝试回答，答对给奖励、答错不给；
- 用强化学习让他逐渐学会"先查哪一页、用哪个公式、什么时候停"。
和传统"背答案"的模型相比，他学会的是工作流程，而不是死记硬背。

### Q68. 这个项目有什么伦理/合规风险？

**答**：
- 金融建议风险：输出是研究辅助，不是投资建议，需要免责声明和人工复核；
- 数据许可：FinQA 来自公开财报，但要确认上游许可和公司合规；
- 数字准确性：模型可能算错，不能直接用于交易/审计；
- 数据泄露：不要把内部未公开财报放进公开 corpus；
- 评估偏差：FinQA 以美国 S&P 500 为主，迁移到其他市场要重新评估。

### Q69. 你和团队/导师意见不一致时怎么处理？

**答（行为题模板）**：先确认目标一致，再把分歧拆成可验证的假设。例如对方认为应该用 7B，我认为 3B 更稳：我会列出显存账本、跑一个 50 update 的 pilot、用数据说话。如果对方坚持，我会执行但记录风险，并设置回滚点。重点不是赢辩论，而是让决策可复盘。

### Q70. 你如何保证代码质量？

**答**：
- 分层：`core.py` 不依赖框架，单测覆盖；
- 60 个 CPU-only 测试，覆盖解析/奖励/BM25/AST/episode/数据；
- 端到端 smoke test（fixtures → 数据 → reward）；
- 训练/评测共用一套逻辑；
- 不修改 upstream，扩展代码独立；
- 所有脚本有 `set -euo pipefail` 和路径自检；
- 文档明确"哪些是估算、哪些是实测"。

### Q71. 如果训练 reward 不上升，你怎么排查？

**答（顺序）**：
1. 看 rollout 输出：有没有 `<answer>`？格式率多少？
2. 看工具：retriever 是否 `no_corpus`/`empty_scope`？calc 是否报错？
3. 看 reward 分布：是否全 0/全 1？组内有没有差异？
4. 看 advantage：是否全 0（全对/全错）？
5. 看 KL/entropy：是否爆炸/塌缩？
6. 看数据：`exe_ans` 是否 NaN、scope 是否匹配；
7. 小实验：固定一条 prompt，手动跑 flow，打印每步 reward。
先排除环境/数据，再怀疑算法。

### Q72. 你下一步的计划？

**答**：
1. 在 4×A40 上跑通 GRPO 1 epoch，拿到 base/GRPO/direct 三组真实数字；
2. 补 RLOO/PPO/Token-GAE 对比；
3. 做上下文和奖励消融；
4. 接稠密检索，比较 BM25；
5. 如果有效，扩到 TAT-QA/FinanceBench 数值子集；
6. 整理成可复现的 report 和开源 repo。

---

## 附录：高频追问速查

| 追问 | 一句话回答 |
|---|---|
| 为什么不用 critic？ | GRPO/RLOO 用组内 baseline，省一个和 actor 同大的网络 |
| 为什么 n=8？ | 组估计稳定性和 rollout 成本的折中 |
| 为什么 max_steps=6？ | FinQA 多数题 2–4 步；6 给纠错余量，太多会拖慢训练 |
| 为什么 response 768？ | 单步 analysis + tool call 通常 <500 token；768 留余量 |
| 为什么 prompt 4096？ | system + 工具 schema + 问题 + 2 条观测，约 2–3K |
| 为什么 KL=0.001？ | 常用起点，防止漂移但不锁死学习 |
| 为什么 lr=1e-6？ | 全参 RL 的保守值；太大容易崩格式 |
| 为什么 batch=32？ | 4×48GB 下 GRPO n=8 的显存/吞吐折中 |
| 为什么 3B？ | 4×48GB 要放 actor/ref/critic/vLLM，3B 无 offload |
| 为什么 BM25？ | 无额外 GPU、可解释、冷启动快；dense 是后续 |
| 为什么 AST 沙箱？ | 防止任意代码执行；只允许算术和少量数学函数 |
| 为什么 reward 只看终止步？ | 结果可验证；step-level shaping 是后续 |
| 为什么报告 unseen？ | 官方 split 是问题级，同文档可能泄漏 |
| 为什么不用 GiGPO？ | 需要 anchor_obs，FinQA 观测几乎不重复 |
| 为什么不用 REMAX？ | 上游 agent flow 未实现，会抛 NotImplementedError |
| 为什么 StepPO 用 gae？ | Agent-R1 没有 steppo estimator，StepPO 通过 step-level gae 暴露 |
| 为什么代码分 upstream/projects？ | 防止扩展污染框架，便于升级和复现 |
| 最大的风险？ | 没有实测结果；文档泄漏；reward hack；7B OOM |
