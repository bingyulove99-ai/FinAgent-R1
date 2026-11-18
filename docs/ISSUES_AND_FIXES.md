# 问题与解决记录（完整版）

本文件按时间顺序记录本次 FinAgent-R1 全流程中遇到的**所有问题、根因与解决办法**，
并给出证据位置。配套实验总结见 `docs/05_experiment_summary_zh.md`。

---

## 1. 旧运行静默续训了已漂移的 checkpoint
- **现象**：旧的 `run_all_trainings.sh` 使用固定 EXP_NAME；verl `resume_mode=auto` 自动从
  `global_step_25`（此前已漂移的策略）继续训练；配置里的 lr 修正被优化器状态覆盖，
  KL 从 3.2 一路涨到 6.4，reward 0.02–0.07，训练无效。
- **根因**：verl 在 `trainer.default_local_dir` 存在 checkpoint 时默认续训；优化器
  `param_groups['lr']` 从 checkpoint 恢复，命令行新 lr 不生效。
- **解决**：
  1. 新的 `run_all_trainings_v2.sh` 为每个算法使用独立输出目录，启动前把陈旧目录移入
     `outputs/_archive/`（绝不静默续训）；
  2. 崩溃重试仅在**同一算法自己的目录**内续训（verl resume）；
  3. `run_all_trainings_v3.sh` 增加 `RESUME=1` 显式续训开关；
  4. 先验证 hydra overrides 与运行日志中的 `actor/lr`。
- **证据**：`logs/train_grpo_collapsed_run1.log`、`logs/metrics_grpo_run2.log`、
  `outputs/_archive/`。

## 2. KL 爆炸 / 奖励刷分（旧配置）
- **现象**：旧 run1/run2 在 ~step 15 后 KL 0.001→4.8→7.3，train reward 0.24→0.05；
  最优 dev 停在 0.2168（step25）。
- **根因**：lr=1e-6 偏大 + kl_coef=0.001 锚定过弱；GRPO std 归一化放大少数幸运样本。
- **解决**：统一改为 lr=5e-7、kl_loss_coef=0.03、low_var_kl、GRPO 使用 **Dr.GRPO**
  （`norm_adv_by_std_in_grpo=False`）。新 GRPO 全程 KL≤0.50，dev 升至 0.2864。
- **证据**：`logs/train_grpo.log`、`logs/devacc_grpo.tsv`。

## 3. 断点续训时 lr 覆盖不生效
- **现象**：同一命令中 `actor_rollout_ref.actor.optim.lr=5e-7`，续训日志仍显示 `actor/lr:1e-6`。
- **根因**：FSDP 恢复 optimizer state 时带回 param_groups 的 lr。
- **解决**：正式实验一律全新开始（步骤 1）；确需续训时接受 checkpoint 中的 lr 或以
  新实验目录重新训练。
- **证据**：`logs/train_grpo.log`（19:42 运行）step 41 `actor/lr:1e-06`。

## 4. PPO 随机初始化 critic 破坏首个 actor 更新
- **现象**：PPO quicktest 中 critic vpred≈3、grad_norm≈4734、vf_loss≈50；
  1 步后 dev 从 0.070 跌至 -0.024。
- **根因**：`critic_warmup=0`，随机 value head 产生巨大偏差；GAE 虽做 masked_whiten，
  但均值偏差仍影响首个更新。
- **解决**：`train_ppo.sh` / `train_token_gae.sh` 设 `trainer.critic_warmup=10`
  （前 10 步只训 critic）；正式 PPO 中 critic vpred 收敛到 0.07。
- **证据**：`logs/ppo_quicktest.log`、`logs/train_ppo.log` step 1-10。

## 5. 评测并发 vLLM 导致 KV cache 显存失败
- **现象**：训练刚结束时启动 4 个评测分片（2 进程/卡，0.40 利用率），3 个报
  `Engine core initialization failed / No available memory for the cache blocks`。
- **根因**：训练进程退出后残留的 vLLM EngineCore/Ray worker 仍占用显存；
  同卡两个 vLLM 同时初始化竞争 KV cache。
- **解决**：加固 `eval_model.sh`：
  1. 评测前 `ray stop --force` + 清理本用户 `VLLM::EngineCore` + 等待 20s；
  2. 4 个分片错峰启动（间隔 25s）；
  3. `wait` 后检查每个分片是否完成（日志含 `avg_retrieved_chunks`）；
  4. 失败/未完成分片逐个**串行**补跑（0.45 利用率），完成后拼接、评分。
- **证据**：`logs/eval_grpo.log`、`logs/eval_token_gae.log`、`eval_model.sh`。

## 6. `eval_model.sh` heredoc 变量展开 bug（误报“评测失败”）
- **现象**：分片与评分全部成功，但脚本最后打印汇总时报 `s: unbound variable` 退出 1 →
  流水线误报“eval failed”。
- **根因**：未加引号的 heredoc 中 `${s}` 被 shell 展开，`set -u` 下报错。
- **解决**：heredoc 改为 `<<'PY'`，通过环境变量传入 RES/TAG；并对空子集（seen_docs=0）
  输出 “(empty split)”。
- **证据**：`logs/eval_rloo.log`、`logs/eval_reinforce.log`（修复前），
  `logs/eval_ppo.log`（修复后正常）。

## 7. `summarize_algo.py` 空子集 KeyError
- **现象**：生成 summary 时报 `KeyError: 'accuracy'`。
- **根因**：官方 FinQA 按文档划分，test 文档全部未见，`seen_docs` 子集为 0 行，
  `aggregate()` 只返回 `{"count": 0}`。
- **解决**：空子集输出 “n/a (empty split)”。
- **证据**：`results/*_summary.md`。

## 8. RLOO / REINFORCE 中后期漂移
- **现象**：
  - RLOO：dev 0.2515(step100) → 0.0685(175)，KL 最高 9.1，熵 0.0007；
  - REINFORCE：dev 0.1827(25) → -0.007(100)，KL 最高 11.6，响应长度暴涨。
- **根因**：3B 模型 + 步级 MDP 下 on-policy 方差大；策略学会“更长的回答/直接作答”
  刷奖励，偏离参考模型。
- **解决**：全 checkpoint 保留（keep=10）+ 硬链接归档；按 **dev 最佳 checkpoint** 合并评测
  （文档 8.5 明确允许并建议报告该规则）。RLOO 取 step100、REINFORCE 取 step25。
- **证据**：`logs/devacc_rloo.tsv`、`logs/devacc_reinforce.tsv`、`results/*_summary.md`。

## 9. PPO 训练期 dev 指标异常（含 critic 算法的验证差异）
- **现象**：同一 base 模型，无 critic 算法初始 dev 为 +0.056~+0.065，而 PPO 为 -0.0254、
  Token-GAE 为 -0.0386；PPO 训练侧熵 ~1.0（其余 ~0.013）。
- **根因**：含 critic 的 rollout/验证路径存在系统性差异（未完全定位；不影响最终统一
  test 推理评估）。
- **解决**：保留原配置完成实验；checkpoint 选择仍用训练期 dev 排序；最终以同一
  `infer_vllm`（贪心）test 评测为准，并在报告中注明该差异。
- **证据**：`logs/train_ppo.log`、`logs/train_token_gae.log`、`results/FINAL_REPORT.md` §6。

## 10. PPO 后期长度刷分崩溃
- **现象**：PPO step150 后响应长度 184→455，dev +0.1009 → -0.1136（step175）。
- **解决**：按 dev 最佳选 step150；最终 test 19.97%。
- **证据**：`logs/train_ppo.log`、`results/ppo_gae_summary.md`。

## 11. checkpoint 轮转风险
- **现象**：默认 `max_actor_ckpt_to_keep=3/4` 会删除早期（可能最优）checkpoint。
- **解决**：设 **10**（保留全部 8 个保存点）；`archive_checkpoints.sh` 每 2 分钟把新
  checkpoint 硬链接到 `outputs/_archive/ckpt/`（同文件系统零额外空间）。
- **证据**：`outputs/_archive/ckpt/`、`logs/archive_checkpoints.log`。

## 12. Token-GAE 长度爆炸 + critic 更新 CUDA OOM，epoch 无法完成
- **现象**：
  1. 训练到 step ~75：响应长度 428→757/768、KL 0.5→5.4（策略退化）；
  2. `critic_update_critic` CUDA OOM（GPU0 44.4GB 容量；torch 33.3GB + vLLM 13.8GB）；
  3. 自动重试从 step 50 续训后，在同一位置再次 OOM；
  4. 手动将 actor/critic token 预算 8192→4096、micro-batch 2→1、vLLM 0.30，
     从 step 50 续训两次，均在 resume 后**首个 critic 更新**处同样 OOM。
- **根因**：退化策略的超长响应使 token 级 critic 的全量 batch（128 轨迹 × ~700 tokens）
  与 critic 全参数 all-gather/优化器状态超出剩余显存；本机仅 2 张 A40 可用
  （GPU1/2 被其他用户占用），无法扩容。
- **解决/结论**：停止重复尝试；保留 checkpoint 25/50 与 `resume_token_gae.sh`；
  按 dev 最佳 **step 50** 合并评测（test acc 10.55%），在最终报告中如实说明
  “Token-GAE 未完成完整 1 epoch（硬件显存限制）”。
- **证据**：`logs/train_token_gae.log`、`logs/train_token_gae.attempt1.log`、
  `logs/train_token_gae_resume.log`、`outputs/finqa_qwen3b_token_gae/selection.txt`。

## 13. 强制停止后残留 Ray/vLLM 进程占用显存
- **现象**：停止训练/评测后 GPU 仍显示 12–43GB 占用，导致后续任务 vLLM 初始化失败。
- **根因**：vLLM V1 的 EngineCore 子进程与 Ray WorkerDict 可能比父进程存活更久。
- **解决**：统一清理流程：
  `ray stop --force` → `pkill -9 -f "[V]LLM::EngineCore"` → `pkill -9 -f "ray::WorkerDict"`
  → 等 10–20s → `nvidia-smi` 复核。已内置在 `eval_model.sh`、`resume_token_gae.sh`。
  ⚠️ 注意 pkill 模式不要匹配到自己的 shell（本次曾两次误杀自身命令，无副作用）。
- **证据**：`eval_model.sh` 清理段、本次会话 `nvidia-smi` 前后对比。

## 14. finalizer 误判训练结束、并发启动基线评测
- **现象**：手动停止 Token-GAE 后，finalizer 的等待循环看到 runner 不存在且当时无
  `infer_vllm`，于是开始跑 base_direct/base_react；与我手动启动的 token_gae 评测并发抢占显存。
- **解决**：立即停止 finalizer 与其基线分片，清理残留显存；Token-GAE 评测完成后按顺序
  重跑 base_react、再执行 `aggregate_results.py`。base_direct 结果已完整保留。
- **证据**：`logs/finalize.log`、`logs/eval_base_react.log`。

## 15. 数据顺序一致性校验
- **现象/风险**：数据集同时存在 parquet 与 jsonl，需保证评测/sample 使用同一顺序。
- **解决**：抽样前对比两者前 3 个 `question_id`（ETR/2016/page_23.pdf-2、
  INTC/2015/page_41.pdf-4、ADI/2011/page_61.pdf-2），完全一致；样例均取前 60/12 条。
- **证据**：`samples/data/test_cases60.jsonl` 与 `samples/predictions/*_cases60.jsonl`
  的 id 一一对应。

---

## 附：经验清单（可复用）

1. 每个算法独立目录 + 启动前归档 + 显式 RESUME 开关，防止静默续训污染。
2. 小 batch 先验证 50 步的 KL/熵/长度趋势，再投入长时间训练。
3. 对长输出任务，总保留全部 checkpoint 并按 dev 最佳选型。
4. 评测前后必须清理残留 vLLM/Ray 进程，分片串行补跑。
5. 所有脚本 `setsid nohup` 运行，配 done/eval_done 标记，可跨会话/断点恢复。
6. 资源受限时优先保证“训练→最佳 checkpoint→统一评测→汇总”链路完整，并如实记录偏差。
