# 结论先行

我能访问并审查公开 GitHub 仓库的主线文件、proposal、retractions、review-stage notes、核心 trainer、configs、eval scripts 与已提交 JSON 结果；但我**没有成功在容器内 clone / 运行仓库**，所以所有“是否可复现”的判断只能基于公开可见代码与日志的静态审计，不能冒充已经本地重跑过。仓库当前公开主线已经从旧的 rho / CSD / AdaBalance 叙事转向 **SPO + Verified Replay CE**，但 paper draft 仍保留旧 claim，且 review-stage 已明确把当前状态评为 **NOT READY**。([GitHub][1])

我的唯一推荐主路线不是“继续包装当前最好的 SPO+Replay n=200 结果”，而是：

> **TRACE-GRPO: Trust-Calibrated Replay and Prompt-Conditioned Credit Assignment for Sparse Binary GRPO**
> 用 **prompt-conditioned posterior / trust gate / replay drift control** 重写现有 verified replay，使 replay 不再是固定 λ、均匀采样、无置信度的成功轨迹 CE，而是按 prompt 难度前沿、成功证据量、staleness、长度、diversity 与训练漂移动态调节。

Confidence: **medium-low**。理由是：仓库中已有正面、负面和不稳定现象足以支持“缺少 trust-calibrated replay control”这个诊断，但目前没有 full-set、多 seed、官方 baseline 和新机制 ablation 证明它一定有效。

---

# Evidence Status Legend

| Label              | Meaning                                                      |
| ------------------ | ------------------------------------------------------------ |
| confirmed evidence | 我直接看到了公开代码、README、proposal、retraction、review note 或 JSON 结果。 |
| likely evidence    | review-stage 或汇总文件声称存在，且与部分 raw JSON 一致，但我未能重跑命令。            |
| hypothesis         | 从现象、代码结构和已有结果反推的方法机制判断。                                      |
| speculation        | 对未来实验可能结果、novelty 强弱、审稿风险的预测。                                |

---

# 第零部分：仓库可读性判断

| Item               |           Found? | Location                                                                               | Notes                                                                                                     |
| ------------------ | ---------------: | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 仓库是否可访问            |              Yes | GitHub root                                                                            | 公开可读，能看到文件树和 raw 文件；但本地 `git clone` 未完成，因此不是动态执行审计。([GitHub][2])                                          |
| 完整代码               |        Partially | `src/`, `scripts/`, `configs/`                                                         | 主线代码可读；但无法保证 Git LFS、未提交日志、wandb、checkpoint 都完整。([GitHub][3])                                             |
| README             |              Yes | `README.md`                                                                            | README 当前 focus 是 **SPO + Verified Replay CE**，并指向 proposal、AUTO_REVIEW、RETRACTIONS。([GitHub][1])         |
| 论文草稿               |              Yes | `paper/main.tex`                                                                       | 存在，但主线仍是旧的 CSD / rho / AdaBalance 叙事，和当前 retraction / README 冲突。([GitHub][4])                             |
| 训练脚本               |              Yes | `scripts/run_aser_mvp.py`, other launch scripts                                        | 当前主线入口是 ASE-R / ASER MVP，支持 `--backbone spo`、`--lambda-rep`、`--pg-weight` 等。([GitHub][5])                 |
| 评估脚本               |              Yes | `scripts/eval_stratified.py`, `launch_fullset_eval.sh`                                 | `eval_stratified.py` 默认评估 first-N GSM8K test；`launch_fullset_eval.sh` 计划 full-set n=1319。([GitHub][6])    |
| configs            |              Yes | `configs/aser_mvp.yaml`, others                                                        | 主 config 记录 Qwen/Qwen3.5-9B、G=2、LoRA、λ replay 等。([GitHub][7])                                             |
| 日志和结果              |    Yes / partial | `results/`                                                                             | 有 wave10/11/13/14 JSON 和 per-seed eval；但 full-set 200-step 总表未看到，部分 provenance 混乱。([GitHub][8])           |
| baseline           | Yes / incomplete | fixed-rho, RFT-only, SPO-only, SFT-gold, base, Dr.GRPO references                      | baseline 存在，但官方 SOTA / mechanism-level baseline 尚未完整复现。([GitHub][9])                                      |
| 失败实验记录             |              Yes | `RETRACTIONS.md`, `review-stage/AUTO_REVIEW.md`, `results/wave14_*`                    | RETRACTIONS 明确撤回旧 rho / ADQ / bandit / exact-rho 等 claim；AUTO_REVIEW 明确列出 fatal weaknesses。([GitHub][10]) |
| ablation           |    Yes / partial | `results/analysis_wave11.json`, `round2_analysis.json`, phase diagram, stratified eval | 有 SPO-only、RFT-only、fixed-rho、λ=0.02、true-dup、phase-grid 等。([GitHub][11])                                 |
| requirements / env | Yes / incomplete | `requirements.txt`                                                                     | 有 Python package requirements；未看到完整 environment lockfile。([GitHub][12])                                   |

## 缺失项

| Missing Item                   | Why Needed                                                        | What I Should Upload                                                         |
| ------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 完整 zip 或可 clone repo snapshot  | 我无法本地 clone / grep / run；需要排除隐藏文件、submodules、未提交脚本差异。             | repo zip，含 `.git` metadata 更好。                                               |
| raw stdout/stderr logs         | JSON 结果不足以证明命令、环境、checkpoint、seed 完全一致。                           | 所有 wave 的 launch logs、slurm logs、wandb/tensorboard export。                   |
| checkpoints / adapter metadata | 需要验证 eval 是否加载正确 checkpoint，是否 stale checkpoint。                  | `checkpoint-final/adapter_config.json`, trainer state, safetensors metadata。 |
| full-set 200-step eval outputs | review note 说 full-set queued，但我未看到完整输出目录。                        | `results/fullset_eval/` 或对应 JSON/CSV。                                        |
| official baseline runs         | NeurIPS 级 claim 需要官方 DAPO / RePO / Dr.GRPO / GSPO 等机制近邻 baseline。 | baseline code commit hash、configs、logs、results。                              |
| intended current paper draft   | `paper/main.tex` 明显 stale；如果另有新稿，需要审查 claim。                      | 最新 PDF / LaTeX / Overleaf export。                                            |

---

# 第一部分：Repository Map

| Component               | Path                                    | Purpose                                          |         Importance | Notes                                                                                                                  |
| ----------------------- | --------------------------------------- | ------------------------------------------------ | -----------------: | ---------------------------------------------------------------------------------------------------------------------- |
| Current README          | `README.md`                             | 标明当前 focus、quick start、项目结构                      |               High | 当前 focus 已改为 **SPO + Verified Replay CE**，旧 claim 需以后续 retraction 为准。([GitHub][1])                                     |
| Active proposal         | `PROPOSAL_SPO_REPLAY.md`                | 当前候选主线、validated claim、weaknesses、file map       |               High | 是当前方法叙事源，但仍只能作为 proposal，不是已证明论文结论。([GitHub][13])                                                                      |
| Retraction log          | `RETRACTIONS.md`                        | 撤回旧结果、confounders、authoritative sources          |           Critical | 必须冻结，作为学术诚信核心文件。([GitHub][10])                                                                                         |
| Auto review             | `review-stage/AUTO_REVIEW.md`           | 内部 reviewer 风险、fatal weaknesses、行动项              |           Critical | 明确指出 n=200 prefix、500-step full-set collapse、SFT-gold crushes、adaptive duplication no effect。([GitHub][9])             |
| Stale paper             | `paper/main.tex`                        | 论文草稿                                             | Critical but stale | 当前主张 CSD/rho/AdaBalance，与 README/RETRACTIONS 冲突，应 archive 或彻底重写。([GitHub][4])                                          |
| Main trainer            | `src/aser_trainer_v14.py`               | ASE-R / SPO / verified replay CE trainer         |           Critical | 当前主线 loss、generation、reward、replay 都在这里。([GitHub][14])                                                                 |
| Prompt stats            | `src/prompt_stats.py`                   | per-prompt EMA baseline / success EMA / hardness |               High | SPO baseline 和 hard prompt 统计基础，但缺 posterior uncertainty / trust。([GitHub][15])                                        |
| Replay bank             | `src/replay_bank.py`                    | 保存 verified successful completions               |               High | 当前均匀 sample，max per prompt，hash dedupe；缺 trust/staleness/frontier。([GitHub][16])                                       |
| Main training script    | `scripts/run_aser_mvp.py`               | 加载 GSM8K train、构造 trainer、保存结果                   |           Critical | 当前训练入口，默认 model Qwen/Qwen3.5-9B，G=2，LoRA，支持 replay flags。([GitHub][5])                                                 |
| Evaluation script       | `scripts/eval_stratified.py`            | greedy eval on GSM8K test                        |           Critical | 文档写明 first-N test evaluation，是当前正面结果最大可靠性问题之一。([GitHub][6])                                                            |
| Full-set eval launcher  | `launch_fullset_eval.sh`                | 计划 n=1319 全测试集评估                                 |               High | 脚本存在，但完整 full-set 输出未看到。([GitHub][17])                                                                                 |
| Main config             | `configs/aser_mvp.yaml`                 | ASER MVP hyperparameters                         |               High | 明确 train split 用于 GRPO，eval 在 GSM8K test；gradient checkpointing false 因 TRL 0.14 rollout corruption note。([GitHub][7]) |
| Wave 11 summary         | `results/analysis_wave11.json`          | 汇总 ASE-R、SPO-only、RFT、fixed-rho                  |               High | 关键 n=200 正面结果来源，但不是 full-set。([GitHub][11])                                                                            |
| Round 2 summary         | `results/round2_analysis.json`          | 进一步汇总 fixed sampler、λ ablation、RFT seeds         |               High | 显示 replay / sampler / λ 敏感性和不稳定性。([GitHub][18])                                                                        |
| Training dynamics       | `results/wave10_training_dynamics.json` | train reward、replay loss、bank size               |             Medium | 支持“训练 reward 改善但不代表泛化”。([GitHub][19])                                                                                  |
| 500-step full-set evals | `results/wave14_500step/evals/*.json`   | 500-step adapters on full GSM8K test             |           Critical | raw JSON 显示 3 seeds mean ≈44.7%，支持 collapse 诊断。([GitHub][20])                                                          |
| Phase diagram           | `results/wave14_phase_diagram/`         | α/β phase sweep                                  |               High | 多个 full-set eval 示例接近 base，说明 scalar phase control 不是核心。([GitHub][21])                                                 |
| Stratified wave13       | `results/stratified_eval_wave13/`       | SFT-gold / true-dup evals                        |               High | SFT-gold seed42 = 84.5%，显著强于当前 RL route。([GitHub][22])                                                                 |

## 必答问题

1. **当前问题**：在 binary reward、small group size，尤其 G=2 的 GRPO/RLVR 设置下，让 math reasoning training 更稳定、更 sample-efficient。README 当前明确 focus 是 SPO + Verified Replay CE。([GitHub][1])
2. **当前已有方法**：`ASERTrainerV14`，使用 SPO per-prompt baseline 加 verified successful completions 的 replay CE。
3. **核心假设**：small-G binary reward 的主要瓶颈是 group 内 all-fail / all-pass 导致 advantage 退化；per-prompt baseline 和 replay verified successes 可以恢复学习信号。
4. **声称解决的 prior limitation**：旧 proposal 认为 fixed-rho / ADQ / bandit 等不可用，SPO+Replay 是更可信的 small-G binary GRPO route。([GitHub][10])
5. **训练入口**：`scripts/run_aser_mvp.py`。
6. **评估入口**：`scripts/eval_stratified.py`，以及 `launch_fullset_eval.sh`。
7. **数据处理**：`scripts/run_aser_mvp.py` 内加载 GSM8K train 并格式化 prompt；`eval_stratified.py` 加载 GSM8K test。([GitHub][5])
8. **模型核心**：Qwen/Qwen3.5-9B + LoRA，配置在 `configs/aser_mvp.yaml`。([GitHub][7])
9. **loss / objective**：`src/aser_trainer_v14.py` 内 `loss_pg + lambda_rep * loss_rep`。([GitHub][14])
10. **baseline**：SPO-only、RFT-only、fixed-rho、SFT-gold、base、Dr.GRPO reference。
11. **configs**：`configs/aser_mvp.yaml` 及 phase configs。
12. **results/logs**：`results/analysis_wave11.json`, `round2_analysis.json`, `wave10_training_dynamics.json`, `wave14_500step`, `wave14_phase_diagram`。
13. **论文 claim**：`paper/main.tex`，但当前不应视为 authoritative。
14. **主线文件**：`README.md`, `PROPOSAL_SPO_REPLAY.md`, `RETRACTIONS.md`, `src/aser_trainer_v14.py`, `src/prompt_stats.py`, `src/replay_bank.py`, `scripts/run_aser_mvp.py`, `scripts/eval_stratified.py`, `configs/aser_mvp.yaml`, `results/analysis_wave11.json`, `review-stage/AUTO_REVIEW.md`。
15. **历史遗留**：rho / CSD / AdaBalance 相关 trainer、paper claim、old research summary。
16. **dead code 候选**：phase α/β routes、adaptive duplication sampler、old rho-grid code；不是删除结果，而是 archive 为 historical evidence。
17. **会影响实验结论的文件**：`eval_stratified.py`, `aser_trainer_v14.py`, `prompt_stats.py`, `replay_bank.py`, result aggregation scripts, config defaults, checkpoint naming/loading scripts。

---

# 第二部分：Result Reliability Audit

| Result ID | Result Name                           | Dataset              | Metric           |                        Claimed Value |                                              Logged Value | Config                 | Seed            | Command                                        | Checkpoint      | Status             | Reliability | Issue                                                                                                       |
| --------- | ------------------------------------- | -------------------- | ---------------- | -----------------------------------: | --------------------------------------------------------: | ---------------------- | --------------- | ---------------------------------------------- | --------------- | ------------------ | ----------- | ----------------------------------------------------------------------------------------------------------- |
| R1        | SPO+Replay / ASE-R n=200              | GSM8K test first 200 | exact answer acc |                          69.4 ± 10.4 |                                    0.6939 std 0.1044, n=9 | `aser_mvp.yaml` likely | 42–51 subset    | likely `run_aser_mvp.py --backbone spo` + eval | wave10 adapters | Partially Verified | medium-low  | Positive but high variance and deterministic first-200 prefix; not paper-grade.([GitHub][11])               |
| R2        | fixed-rho 0.70 n=200                  | GSM8K first 200      | acc              |                         ~54.2 / 53.7 |                                    0.5367 std 0.1051, n=9 | mixed prior/wave       | 42–51 subset    | unclear exact                                  | mixed           | Partially Verified | medium-low  | Baseline provenance mixed; useful but not final.([GitHub][11])                                              |
| R3        | SPO-only n=200                        | GSM8K first 200      | acc              |                                ~52.2 |                                    0.5217 std 0.1097, n=3 | aser no replay         | 42–44           | likely `--lambda-rep 0`                        | wave            | Partially Verified | medium-low  | Only 3 seeds; shows SPO alone insufficient.([GitHub][11])                                                   |
| R4        | RFT-only n=200                        | GSM8K first 200      | acc              |                                ~36.1 |                                 round2 n=7 around 35–38.5 | ASER pg off?           | mixed           | unclear                                        | mixed           | Missing Command    | low         | Analysis label inconsistency: one file says `rft_only (n=3)` but `n_seeds=1`; round2 has n=7.([GitHub][11]) |
| R5        | Wave10 training dynamics              | GSM8K train          | reward last50    |       SPO+Replay train reward higher |            spo_full mean last50 ≈0.724 vs spo_only ≈0.535 | wave10                 | multiple        | training script                                | adapters        | Verified as logged | medium      | Train reward does not prove test generalization.([GitHub][19])                                              |
| R6        | 500-step SPO+Replay full-set collapse | GSM8K test full 1319 | acc              |                     review says 44.6 | seed42 0.5383, seed43 0.4223, seed44 0.3791; mean ≈0.4466 | wave14_500step         | 42–44           | unknown                                        | final adapters  | Verified as logged | medium      | Strong negative signal; only 3 raw eval seeds inspected, but matches AUTO_REVIEW.([GitHub][20])             |
| R7        | Phase diagram α/β full-set            | GSM8K full 1319      | acc              | review says no effect, near base 25% |                           inspected examples ≈0.263–0.267 | phase grid             | seed42 examples | launch grid                                    | phase adapters  | Partially Verified | medium-low  | Single seed examples; enough to de-prioritize scalar α/β as main mechanism.([GitHub][23])                   |
| R8        | SFT-gold n=200                        | GSM8K first 200      | acc              |                           84.6 ± 0.6 |                                              seed42 0.845 | wave13                 | seed42 visible  | unknown                                        | adapter         | Partially Verified | medium      | Strong upper-bound/simple baseline; current method far below.([GitHub][22])                                 |
| R9        | true-dup n=200                        | GSM8K first 200      | acc              |                                mixed |                   0.71, 0.54, 0.685, 0.82 for seeds 42–45 | wave13                 | 42–45           | unknown                                        | adapters        | Partially Verified | low-medium  | High variance; duplicate scheduling alone unstable.([GitHub][24])                                           |
| R10       | Old rho / ADQ / CSD claims            | GSM8K / old configs  | various          |                older positive claims |                                                 retracted | old                    | mixed           | mixed                                          | mixed           | Contradicted       | unusable    | RETRACTIONS says confounded / superseded; cannot support new paper.([GitHub][10])                           |

**Audit verdict:** 当前最可信的结论不是“ASE-R 已经有效”，而是：**SPO+Replay 在 deterministic n=200 prefix 有正面 signal，但它对 full-set / longer horizon / seed stability 尚未成立；旧 rho/CSD 叙事不可用；固定 replay 可能是短期有效、长期漂移的局部 trick。**

---

# 第三部分：代码正确性审查

| Priority | File                         | Function/Class                    | Code Region                                         | Suspicion                                                                                                               | Evidence                                                                                                                   | How to Verify                                                                               | Proposed Fix for Claude Code                                                                                                                          | Expected Effect                                     | Confidence             |
| -------: | ---------------------------- | --------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------- |
|       P0 | `scripts/eval_stratified.py` | `load_gsm8k_test`, main eval loop | first-N selection and `accuracy = correct / args.n` | 正面结果使用 deterministic first 200 test prefix，不能代表 full test；若 n 超过 loaded length，denominator 也可能错                         | docstring says evaluates first N; code selects `range(n)`; AUTO_REVIEW flags prefix issue.([GitHub][6])                    | Add test asserting selected ids for `first_n`, `full`, `random`; compare n=200 vs full 1319 | Add explicit `--selection {first_n,full,random,stratified}`; default for paper eval must be `full`; accuracy denominator `len(ds)`; save question ids | Prevents cherry-pick / prefix artifact              | high                   |
|       P0 | `src/aser_trainer_v14.py`    | `get_per_token_logps`             | model forward                                       | `attention_mask` not passed despite left padding; logprobs / KL / loss may be polluted by pad context                   | trainer constructs `attention_mask`, but helper calls `mdl(input_ids, num_logits_to_keep=...)` without mask.([GitHub][14]) | Unit test with left-padded prompts: compare logprobs with/without mask                      | Change signature to pass `attention_mask`; use it for policy and ref model                                                                            | More correct gradients/KL; may change prior results | medium-high            |
|       P1 | `src/aser_trainer_v14.py`    | reward decoding                   | completion decode                                   | Reward may decode tokens after EOS because `completion_ids` decoded directly while `completion_mask` is only used later | code builds `completion_mask` after EOS but reward completion text comes from raw completion ids.([GitHub][14])            | Construct completion with correct answer before EOS and junk after EOS; check reward        | Decode only valid tokens up to EOS or mask after EOS to pad before decoding                                                                           | Cleaner reward labels, less noise                   | medium                 |
|       P1 | `src/replay_bank.py`         | `sample`                          | uniform replay sampling                             | Uniform successful replay lacks trust, age, prompt frontier, length guard; can overfit or amplify stale traces          | ReplayBank stores unique successes and random samples over all items.([GitHub][16])                                        | Log replay age/prompt distribution/length; correlate with full-set collapse                 | Replace with trust-gated weighted sampler in new method                                                                                               | Main mechanism fix                                  | high as mechanism flaw |
|       P1 | `src/prompt_stats.py`        | `PromptStatsStore`                | baseline/hardness                                   | Per-prompt EMA lacks uncertainty/evidence count semantics; unseen hardness defaults may distort hard sampling           | `get_hardness` returns max for unseen; baseline is simple EMA.([GitHub][15])                                               | Log seen counts; AUTO_REVIEW says SPO baseline barely persistent                            | Rewrite to posterior state with counts, success/fail, uncertainty, saturation                                                                         | Enables prompt-conditioned credit                   | medium-high            |
|       P1 | `src/aser_trainer_v14.py`    | replay CE                         | tokenization / labels                               | Replay CE uses fixed λ and token-level CE; may dominate or bias by length without adaptive token budget                 | loss is `loss_pg + lambda_rep * loss_rep`, replay bank grows to hundreds/thousands.([GitHub][14])                          | Log replay token ratio, length vs reward, grad norm PG vs CE                                | Add λ_eff, token budget cap, length guard, replay contribution logging                                                                                | Prevent replay drift                                | medium                 |
|       P2 | `scripts/analyze_wave11.py`  | result aggregation                | labels / n_seeds                                    | Provenance/labels inconsistent for RFT-only                                                                             | analysis JSON says label `rft_only (n=3)` but n_seeds=1; round2 says n=7.([GitHub][11])                                    | Rebuild table from per-run manifests                                                        | Enforce result schema: run_id, seed, config hash, checkpoint, eval ids                                                                                | Cleaner reliability audit                           | high                   |
|       P2 | `scripts/run_aser_mvp.py`    | config override                   | CLI/config interaction                              | Config may not capture all CLI overrides; exact command provenance missing                                              | script accepts many flags and writes JSON summaries, but visible summaries lack full command/checkpoint hash.([GitHub][5]) | Compare saved config to actual args                                                         | Save full resolved config, git hash, command, env, dataset ids                                                                                        | Reproducibility                                     | medium                 |
|       P2 | `scripts/run_aser_mvp.py`    | duplication sampler               | sampler monkey patch                                | Adaptive duplication / true-dup effects unstable or no effect; may be dead route                                        | AUTO_REVIEW says adaptive duplication no effect; true-dup per-seed unstable.([GitHub][9])                                  | Run sampler diagnostic: actual duplicated prompt ratio per batch                            | Freeze as ablation; do not use as main                                                                                                                | Avoid false mechanism                               | medium                 |

**Important contamination note:** P0 eval prefix and P0/P1 trainer issues mean current n=200 positive should be treated as **signal**, not proof. They can motivate mechanism design, but cannot be used as strong evidence for paper claims.

---

# 第四部分：Claim-Code-Result Matrix

| Claim                                                         | Source File                           | Implementation File                       | Result Evidence                                            | Status                             | Problem                                                                                                                    | Confidence |
| ------------------------------------------------------------- | ------------------------------------- | ----------------------------------------- | ---------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------- |
| CSD / rho / AdaBalance stability map is the main contribution | `paper/main.tex`                      | old rho / AdaBalance code                 | RETRACTIONS and README supersede it                        | Contradicted                       | Paper is stale; submitting this would violate project’s own retractions.([GitHub][4])                                      | high       |
| SPO + Verified Replay CE is current focus                     | `README.md`, `PROPOSAL_SPO_REPLAY.md` | `src/aser_trainer_v14.py`                 | analysis_wave11 positive n=200                             | Partially Supported                | Supported only on first-200 prefix; full-set longer-horizon result contradicts stability.([GitHub][1])                     | medium     |
| SPO+Replay is stable binary-reward GRPO at small G            | proposal / README wording             | `ASERTrainerV14`                          | n=200 mean high but std 10.4; 500-step full-set mean ~44.6 | Partially Supported / Contradicted | “stable” is too strong; high variance and collapse.([GitHub][11])                                                          | high       |
| Adaptive duplication helps                                    | old routes / run flags                | sampler monkey patch in `run_aser_mvp.py` | AUTO_REVIEW says no effect; true-dup unstable              | Unsupported                        | Should not be main claim; only ablation/historical evidence.([GitHub][9])                                                  | medium     |
| RFT-only is insufficient                                      | round2 / analysis                     | `pg_weight=0` path likely                 | RFT-only around 35–38%                                     | Supported as negative evidence     | Command provenance imperfect, but repeated low values support this.([GitHub][18])                                          | medium     |
| SFT-gold is much stronger than RL route                       | AUTO_REVIEW / wave13                  | SFT baseline script not fully inspected   | seed42 84.5%, review says 84.6±0.6                         | Supported                          | This constrains claims; method cannot pretend weak baselines are enough.([GitHub][22])                                     | medium     |
| Train/test split is clean                                     | AUTO_REVIEW                           | data loading scripts                      | Review asserts clean split                                 | Partially Supported                | Static code uses train for training/test for eval, but exact dataset/version/provenance still needs manifest.([GitHub][7]) | medium     |
| α/β phase diagram provides stable control                     | phase configs / proposal remnants     | `ASERTrainerV14` α/β flags                | phase full-set examples near base                          | Unsupported                        | Scalar phase control appears non-mechanistic.([GitHub][23])                                                                | medium     |

---

# 第五部分：Phenomenon Ledger

| ID  | Observation                                                      | Type                      | Where Found                      | Setting                  | Metric          | Compared To            | Reliability | What It Suggests                                                | What It Rules Out                               | Confidence |
| --- | ---------------------------------------------------------------- | ------------------------- | -------------------------------- | ------------------------ | --------------- | ---------------------- | ----------- | --------------------------------------------------------------- | ----------------------------------------------- | ---------- |
| P1  | SPO+Replay reaches ~69.4% on first 200 GSM8K, n=9, but std ~10.4 | Positive + unstable       | `analysis_wave11.json`           | 200-step, first 200 test | acc             | fixed-rho/SPO-only/RFT | medium-low  | SPO+Replay contains real short-horizon signal                   | Does not prove full-set stability               | medium     |
| P2  | 500-step full-set seeds collapse to ~53.8/42.2/37.9              | Negative                  | `wave14_500step/evals`           | full 1319 test           | acc             | n=200 positive         | medium      | Fixed replay / training horizon causes drift or overfit         | Rules out “just train longer”                   | high       |
| P3  | SFT-gold ~84.5% on first 200                                     | Negative constraint       | wave13 / AUTO_REVIEW             | first 200                | acc             | SPO+Replay             | medium      | Strong supervised/replay baseline is far ahead                  | Cannot claim SOTA or strong empirical dominance | high       |
| P4  | Phase α/β grid examples near ~26% full-set                       | Negative                  | `wave14_phase_diagram`           | full-set, seed42         | acc             | base ~25.5             | medium-low  | Scalar advantage weighting not the missing mechanism            | Rules out global α/β as main route              | medium     |
| P5  | RFT-only around 35–38%                                           | Negative                  | `round2_analysis.json`           | first 200                | acc             | SPO+Replay             | medium-low  | CE/replay-only imitation is insufficient                        | Rules out pure replay/RFT as main               | medium     |
| P6  | SPO-only ~52.2%, below SPO+Replay but above RFT                  | Mixed                     | `analysis_wave11.json`           | first 200                | acc             | RFT/fixed-rho          | medium-low  | Per-prompt baseline helps but needs positive memory             | Rules out SPO-only as final method              | medium     |
| P7  | λ=0.02 gives 0.75 and 0.39 in two seeds                          | Unstable                  | `round2_analysis.json`           | first 200                | acc             | λ=0.05                 | low-medium  | Replay pressure is highly sensitive and needs gating            | Rules out fixed scalar λ                        | medium     |
| P8  | true-dup seeds vary 0.54–0.82                                    | Unstable                  | wave13 per-seed JSON             | first 200                | acc             | no-dup                 | low-medium  | Prompt scheduling alone is unstable                             | Rules out duplication-only mechanism            | medium     |
| P9  | Prompt baseline “barely persistent” in review                    | Anomalous                 | AUTO_REVIEW                      | training                 | prompt coverage | method assumption      | medium      | Per-prompt stats too sparse for naïve EMA                       | Rules out simple EMA hardness as enough         | medium     |
| P10 | Train reward improves while full-set eval poor                   | Anomalous                 | training dynamics vs wave14 eval | train vs test            | reward/acc      | generalization         | medium      | Objective aligns with training successes but not generalization | Rules out train reward as evidence              | high       |
| P11 | Old rho/ADQ/bandit/exact-rho claims retracted                    | Negative integrity        | RETRACTIONS                      | old experiments          | various         | current route          | high        | Project has history of confounded mechanisms                    | Rules out using old positives as evidence       | high       |
| P12 | Evaluation first-N prefix                                        | Anomalous / contamination | `eval_stratified.py`             | n=200 eval               | acc             | full-set               | high        | Current positive subset can be misleading                       | Rules out n=200 prefix-only paper claim         | high       |

---

# 第六部分：Design Constraints

| Constraint ID | Derived From Observation | Constraint Type    | Meaning                                                   | Implication for New Method                                                            | Confidence  |
| ------------- | ------------------------ | ------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------- |
| D1            | P1, P6                   | Must Preserve      | SPO-style per-prompt baseline has useful signal           | Keep SPO as base credit assignment, but do not make it final method                   | medium      |
| D2            | P1, P5                   | Must Preserve      | Verified successes are valuable, but not sufficient alone | Keep replay only as controlled evidence, not uniform CE                               | medium      |
| D3            | P2, P7                   | Must Fix           | Fixed replay pressure drifts or overfits                  | Add adaptive per-prompt replay coefficient and drift guard                            | high        |
| D4            | P4                       | Must Avoid         | Global scalar α/β tuning does not solve mechanism         | Archive phase diagram as negative evidence                                            | medium      |
| D5            | P8, P9                   | Must Stabilize     | Hard prompt duplication / sparse prompt EMA are unstable  | Use posterior uncertainty and evidence count, not raw hardness                        | medium      |
| D6            | P10                      | Must Control       | Train reward can diverge from test generalization         | Add full-set gate, heldout proxy, length/KL/replay contribution logs                  | high        |
| D7            | P12                      | Must Control       | Deterministic prefix eval can create false positives      | Paper evaluation must be full-set or preregistered random/stratified subset           | high        |
| D8            | P3                       | Must Differentiate | Strong SFT baseline dominates                             | Method must beat strong fair baselines or narrow claim to data-efficient RL mechanism | high        |
| D9            | P11                      | Must Not Claim     | Old rho/CSD claim contaminated                            | Remove old claims from paper and README main narrative                                | high        |
| D10           | P1–P12                   | Must Test          | New route must prove it is not old best fragment          | Run A/B/C: existing best only, new infra without mechanism, full new method           | high        |
| D11           | Code audit               | Must Fix           | attention_mask / EOS / provenance bugs may affect results | Fix before interpreting new experiments                                               | medium-high |
| D12           | Related work             | Must Differentiate | replay-enhanced RL already exists                         | Novelty must be trust-calibrated prompt-conditioned replay, not “we add replay”       | high        |

---

# 第七部分：Negative-to-Insight Analysis

| Negative Observation       | Failed Assumption                                            | Why the Assumption Failed                                                            | What Mechanism Is Missing                  | New Design Requirement                                                    |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------------------------- |
| 500-step full-set collapse | More training with SPO+Replay should improve or stabilize    | Fixed replay can amplify stale, narrow, prompt-specific successes                    | Drift-controlled replay trust              | Replay must decay, cap, and be monitored by full-set / proxy drift        |
| Phase diagram near base    | Scalar α/β advantage shaping captures the right control knob | Failure is prompt- and evidence-dependent, not global scalar                         | Prompt-conditioned credit assignment       | Replace scalar phase route with per-prompt posterior/frontier             |
| RFT-only weak              | Verified answer imitation is enough                          | CE without on-policy reward credit copies narrow traces and lacks exploration signal | Coupled PG + controlled replay             | Keep PG as primary, replay as auxiliary with adaptive λ                   |
| SPO-only limited           | Per-prompt baseline fixes small-G degeneracy                 | It restores some signal but lacks memory of rare successes                           | Verified positive memory, but calibrated   | Use replay as memory, not as uniform loss                                 |
| λ=0.02 unstable            | Replay coefficient is a tunable constant                     | Same λ can underfit one seed and overfit another                                     | Adaptive λ by prompt trust and uncertainty | λ must depend on prompt state, replay item state, and training time       |
| true-dup high variance     | Duplicating hard prompts creates stable learning             | “Hardness” is under-measured; unseen or sparse prompts are noisy                     | Uncertainty-aware frontier selection       | Only upweight prompts with enough evidence and non-saturated posterior    |
| SFT-gold crushes RL route  | Weak RL baselines are enough                                 | Supervised trace data is a much stronger baseline                                    | Strong baseline discipline                 | Include SFT-gold / RePO / DAPO / Dr.GRPO comparisons before strong claims |
| n=200 prefix issue         | First 200 test questions approximate benchmark               | Deterministic prefix can be unrepresentative                                         | Evaluation provenance control              | Save eval ids, use full-set, report all seeds                             |

---

# 第八部分：Method Synthesis Table

| Evidence Fragment     | Source in Repo                       | What It Reveals                            | Generalized Principle                      | Use in New Method? | How to Transform It                                                      |
| --------------------- | ------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------ | ------------------------------------------------------------------------ |
| SPO baseline          | `PromptStatsStore`, `ASERTrainerV14` | Per-prompt baseline helps small-G credit   | Non-degenerate prompt credit is useful     | Yes                | Keep as base advantage, but add posterior confidence                     |
| Verified Replay CE    | `ReplayBank`, `ASERTrainerV14`       | Successful trajectories are valuable       | Rare positives should be remembered        | Yes                | Replace uniform replay with trust-gated, prompt-conditioned replay       |
| RFT-only failure      | `round2_analysis.json`               | CE alone is weak                           | Replay needs on-policy anchoring           | Yes                | Keep PG primary; replay auxiliary only                                   |
| λ instability         | `round2_analysis.json`               | Fixed replay strength is brittle           | Replay pressure must be adaptive           | Yes                | Compute λ_eff per prompt/item                                            |
| 500-step collapse     | wave14 evals                         | Short-term gains can become drift          | Need training-time drift control           | Yes                | Add age decay, token budget, checkpoint gates, full-set eval checkpoints |
| Phase diagram failure | wave14 phase grid                    | Global scalar shaping not enough           | Mechanism must be local and data-dependent | No as main         | Archive phase route; keep as negative evidence                           |
| SFT-gold strong       | wave13                               | Simple supervised baseline is hard to beat | Need honest baseline hierarchy             | Yes as baseline    | Keep only as strong baseline / upper-bound reference                     |
| adaptive duplication  | run flags / true-dup                 | Scheduling alone unstable                  | Hardness must be calibrated                | Maybe ablation     | Replace raw duplication with frontier score                              |
| old CSD/rho claims    | paper/retractions                    | Confounded positives can mislead           | Claim hygiene is essential                 | No                 | Archive and cite as retracted internal route                             |

---

# 第九部分：Missing Mechanism Diagnosis

1. **Missing Mechanism Name:**
   **Prompt-conditioned trust-calibrated replay credit with drift control**

2. **One-Sentence Diagnosis:**
   当前方法把 verified success 当成等价、可信、可无限复用的 CE 目标，却没有判断“哪个 prompt 的哪个成功轨迹在当前训练阶段值得 replay、该 replay 多强、何时过期、是否正在造成 full-distribution drift”。

3. **Evidence From Positive Results:**
   SPO+Replay first-200 n=200 的正面 signal 说明“per-prompt baseline + positive memory”确实可能比 fixed-rho、SPO-only、RFT-only更有用。([GitHub][11])

4. **Evidence From Negative Results:**
   500-step full-set collapse 和 phase diagram near-base 说明固定 replay / scalar control 没有解决泛化和长程稳定性。([GitHub][20])

5. **Evidence From Unstable Results:**
   λ=0.02 两个 seed 分裂、true-dup 高 variance、SPO+Replay std 10.4，都指向缺少 per-prompt / per-item confidence calibration。([GitHub][11])

6. **Evidence From Failed Ablations:**
   RFT-only 不足说明 replay CE 不是独立机制；SPO-only 不足说明 baseline 也不是独立机制；phase α/β 不足说明全局调参不是机制。

7. **Why Existing Method Cannot Solve It:**
   `ReplayBank.sample` 是随机均匀采样，`PromptStatsStore` 只有 EMA baseline / success EMA，`ASERTrainerV14` 只有固定 `lambda_rep`，没有 posterior uncertainty、staleness、frontier、length guard、per-prompt λ_eff 或 drift gate。([GitHub][14])

8. **Why Simple Tuning Cannot Solve It:**
   λ 和 α/β 的不稳定/无效已经说明问题不是一个全局 scalar 最优值，而是 replay 何时可信、对哪个 prompt 有益、是否 stale 的结构问题。

9. **Why Existing Best Positive Fragment Is Insufficient:**
   现有 best fragment 是固定 SPO+Replay；它解释不了 500-step collapse、phase null、true-dup variance 和 SFT-gold gap。

10. **What New Mechanism Must Do:**
    它必须用 prompt posterior、success evidence、uncertainty/frontier、age decay、diversity/length guard 和 replay token budget 动态控制 replay；并用 full-set / stratified eval 和 replay-drift logging 证明机制真的发生。

11. **Confidence:**
    **medium-low**。现象支持诊断，但新机制尚未实现和验证。

---

# 第十部分：New MAIN METHOD PATH

## New MAIN METHOD PATH

1. **Method Name Placeholder:**
   **TRACE-GRPO** — Trust-Calibrated Replay and Prompt-Conditioned Credit Assignment for Sparse Binary GRPO.

2. **One-Sentence Core Idea:**
   在 SPO 小组 binary GRPO 上，把 verified replay 从固定 λ 的均匀 CE，改成由 prompt posterior、frontier uncertainty、success trust、staleness 和 drift budget 控制的 adaptive replay credit。

3. **Core Missing Mechanism It Adds:**
   **Prompt-conditioned trust gate + adaptive replay coefficient + drift-aware replay budget.**

4. **What Phenomena It Explains:**
   它解释 SPO+Replay first-200 为什么有 signal：rare verified successes 对某些 prompts 有价值；也解释 500-step full-set 为什么 collapse：成功轨迹被无置信度、无过期机制地均匀 replay，导致 narrow overfitting 或 distribution drift。

5. **What Negative Results It Fixes:**
   Fixed λ instability、phase α/β null、RFT-only weak、SPO-only incomplete、true-dup unstable、train reward 与 test gap。

6. **What Existing Positive Signals It Generalizes:**
   它继承“per-prompt baseline”和“verified success memory”，但把它们提升为“prompt posterior credit + trust-calibrated replay”。

7. **Why Existing Best Path Is Not Enough:**
   Existing best path 只是在 fixed SPO+Replay 上得到 n=200 prefix gain；它没有回答 replay 何时应该关闭、降低、过期、按 prompt 改变，也没有 full-set stability 机制。

8. **Core Mechanism:**
   对每个 prompt `x` 维护 posterior state：成功/失败 count、EMA baseline、success probability、uncertainty、saturation、recent improvement、replay exposure。对每个 replay item `(x, y+)` 维护 trust state：verified reward、age、length、hash diversity、source checkpoint、prior replay count。训练时计算
   `λ_eff(x,y,t) = λ_max * frontier(x,t) * trust(x,y,t) * drift_budget(t)`。

9. **New Objective / Loss:**
   `L_total = L_SPO_PG + E[λ_eff(x,y,t) * CEπ(y+|x)] + β_KL L_KL + η L_budget`。
   `L_budget` 可作为 soft penalty，使 replay CE token contribution 不超过目标比例。

10. **New Architecture or Module:**
    无需新 backbone；新增 `PromptCreditState`, `TrustGatedReplayBank`, `ReplayDriftMonitor`, adaptive λ logging。

11. **New Training Procedure:**
    每 step 生成 G=2 rollouts，计算 reward 和 SPO advantage；更新 prompt posterior；只将 verified successes 写入 trust replay bank；按 frontier/trust 采样 replay；按 λ_eff 加权 CE；记录 replay contribution、posterior distribution、drift metrics。

12. **New Evaluation Protocol:**
    禁止只用 deterministic first-200 作主结果；必须 full GSM8K test n=1319，多 seed，保存 eval ids，并包含 A/B/C mechanism ablation。

13. **What Existing Components It Reuses:**
    Qwen/Qwen3.5-9B setup、LoRA config、SPO advantage idea、verified reward extraction、basic training/eval scaffolding。

14. **What Existing Components It Deletes:**
    删除/停止使用 old CSD/rho/AdaBalance paper claim；phase α/β 不再主线。

15. **What Existing Components It Rewrites:**
    `replay_bank.py`, `prompt_stats.py`, `aser_trainer_v14.py` 或 fork 成 `trace_grpo_trainer.py`, `eval_stratified.py`, result aggregation.

16. **What Existing Components It Keeps Only as Ablation:**
    fixed SPO+Replay, SPO-only, RFT-only, true-dup/adaptive duplication, phase α/β.

17. **What Existing Components It Keeps Only as Baseline:**
    SFT-gold, base, fixed-rho, Dr.GRPO / GRPO.

18. **Why This Is Not Merely the Existing Best Path:**
    Existing best path = constant `lambda_rep=0.05` + uniform replay. TRACE-GRPO 的核心实验对象是 `λ_eff` 的 prompt-conditioned trust gate；如果把 gate 关掉，它退化为 existing best fragment，所以可以被 A/B/C ablation 直接检验。

19. **Why This Could Produce Real Positive Results:**
    它保留了现有短期正面 signal 的来源，但专门修复导致不稳定和 collapse 的 replay 信任/漂移缺口。

20. **Why This Is Mechanism-Level Different from Prior Work:**
    它不是泛泛“加 replay”；区别必须落在 binary small-G GRPO 下的 prompt posterior、verified success trust、frontier uncertainty 和 drift budget。注意：这点有 novelty risk，因为 RePO 已经提出 replay-enhanced policy optimization。([arXiv][25])

21. **Main Risk:**
    RePO / DAPO / GSPO 等机制近邻可能覆盖大部分 novelty；另外 full-set 结果可能仍不超过 SFT-gold 或 strong official baselines。

22. **Minimal Falsification Experiment:**
    在 GSM8K full test n=1319、seeds 42/43/44 上比较：
    A. Existing Best Positive Fragment Only；
    B. New infrastructure but no trust gate；
    C. Full TRACE-GRPO。
    如果 C 不显著优于 A/B，且 500-step 仍 collapse，则 TRACE route falsified or needs pivot.

23. **Confidence:**
    **medium-low**，但这是目前最符合现象库的唯一主路线。

---

# 第十一部分：Formal Method Description

## 1. Problem Setup

给定 prompts `x ∈ D_train`，policy `πθ`，binary reward `r(x,y) ∈ {0,1}`。每个 step 对 prompt `x` 采样 `G=2` completions。目标是在 sparse binary reward 下提高 math answer accuracy，同时避免 small-group GRPO 的 all-fail/all-pass degeneracy 和 replay-induced drift。

## 2. Existing Method Failure

当前 `ASERTrainerV14` 使用 SPO baseline 和 fixed verified replay CE：

`L ≈ L_pg + λ_rep L_replay`

但 `λ_rep` 是全局固定值，replay sampling 近似均匀，prompt stats 是简单 EMA。它无法判断某个 replay item 是否 stale、是否过度 replay、是否只对 prefix subset 有效。([GitHub][14])

## 3. New Insight

Sparse binary GRPO 的关键不是“有没有成功样本”，而是：

> **成功样本什么时候可信、对哪个 prompt 仍处在学习前沿、应该以多大权重 replay，以及什么时候 replay 已经开始制造 drift。**

## 4. Method Overview

新增三个状态模块：

1. `PromptCreditState(x)`：posterior success state。
2. `TrustGatedReplayBank`：保存 verified successes 并计算 item trust。
3. `ReplayDriftMonitor`：控制 replay CE token budget、length drift、KL drift、full-set / proxy eval gate。

## 5. Algorithm

**Algorithm: TRACE-GRPO**

**Input:**
policy `πθ`, reference policy `πref`, train prompts `D`, reward function `R`, group size `G=2`, max replay coefficient `λ_max`, posterior prior `(α0, β0)`, replay budget `ρ_rep`.

**Output:**
trained adapter, prompt credit states, replay bank, full provenance logs.

**Steps:**

1. Sample minibatch prompts `x_b`; generate `G` completions `y_{b,j}` from `πθ`.
2. Compute rewards `r_{b,j} = R(x_b, y_{b,j})`; compute SPO advantage
   `A_{b,j} = r_{b,j} - b(x_b)`.
3. Update `PromptCreditState(x_b)` with success/failure counts, EMA baseline, posterior success estimate, uncertainty, saturation and replay exposure.
4. Insert verified successes `r=1` into `TrustGatedReplayBank` with prompt id, token ids, length, source step, hash, reward, and replay count.
5. For replay samples `(x,y+)`, compute
   `frontier(x,t)`, `trust(x,y,t)`, `drift_budget(t)`, then
   `λ_eff = clip(λ_max * frontier * trust * drift_budget, 0, λ_max)`.
6. Optimize
   `L_total = L_SPO_PG + L_TRACE_Replay + β_KL L_KL + η L_budget`.
7. Log `λ_eff` distribution, replay age, replay token ratio, prompt posterior histogram, all-pass/all-fail rates, length/reward correlation, KL, reward, eval ids.
8. Save checkpoint only with full run manifest and config hash.

## 6. Objective

For prompt `x`, posterior estimate:

[
\hat p_x = \frac{\alpha_x}{\alpha_x+\beta_x}
]

Evidence confidence:

[
c_x = \min(1, n_x / n_{\min})
]

Frontier score:

[
F_x = 4 \hat p_x (1-\hat p_x) \cdot c_x
]

This is high for prompts that are neither solved nor hopeless, but only after enough evidence.

Replay item trust:

[
T_{x,y,t} = \mathbf{1}[R(x,y)=1]\cdot \exp(-(t-s_y)/\tau)\cdot D_{x,y}\cdot L_{x,y}\cdot (1 - S_x)
]

where `s_y` is source step, `D` is diversity/dedup factor, `L` is length guard, and `S_x` is saturation / overexposure penalty.

Adaptive replay coefficient:

[
\lambda_{x,y,t} = \mathrm{clip}(\lambda_{\max} F_x T_{x,y,t} B_t, 0, \lambda_{\max})
]

where `B_t` is drift budget from replay contribution, KL, length, and validation/full-set proxy monitors.

Policy loss:

[
L_{\mathrm{SPO}} =
-\mathbb{E}*{x,y\sim\pi*\theta}
\left[
A(x,y) \sum_t \log \pi_\theta(y_t|x,y_{<t})
\right]
]

Replay loss:

[
L_{\mathrm{TRACE}} =
\mathbb{E}*{(x,y^+)\sim \mathcal{B}}
\left[
\lambda*{x,y,t}
\cdot
\mathrm{CE}*{\pi*\theta}(y^+|x)
\right]
]

Total:

[
L_{\mathrm{total}} =
L_{\mathrm{SPO}}
+
L_{\mathrm{TRACE}}
+
\beta_{\mathrm{KL}}L_{\mathrm{KL}}
+
\eta L_{\mathrm{budget}}
]

## 7. Training Pipeline

* Existing variables reused: prompt ids, rewards, completion ids, completion mask, prompt_stats baseline, replay_bank entries.
* New variables added: `alpha_x`, `beta_x`, `n_x`, `frontier_x`, `trust_xy`, `lambda_eff`, `replay_age`, `replay_exposure`, `drift_budget`.
* Required new logging: per-step mean/percentiles of `λ_eff`, replay contribution ratio, posterior histogram, replay item age, full-set eval checkpoints.

## 8. Inference / Evaluation Pipeline

* Greedy evaluation remains acceptable for consistency, but must run on full GSM8K test by default.
* Save exact eval question ids, dataset version, adapter path, checkpoint step, generation args.
* Report mean ± std / CI across seeds, not best seed.

## 9. Expected Empirical Signature

TRACE-GRPO should show:

1. Same or better n=200 first subset than fixed SPO+Replay.
2. Better full 1319 GSM8K accuracy than A/B controls.
3. Lower seed variance.
4. No 500-step collapse relative to 200-step.
5. Replay `λ_eff` should concentrate on frontier prompts, not all successes.
6. Replay age and exposure should be bounded.

## 10. Required Ablations

* A. Existing fixed SPO+Replay.
* B. TRACE infrastructure with `λ_eff = constant`.
* C. Full TRACE-GRPO.
* Remove frontier only.
* Remove trust age decay only.
* Remove drift budget only.
* SPO-only.
* RFT-only.
* SFT-gold.
* Official RePO / DAPO / Dr.GRPO if feasible.

---

# 第十二部分：Related Work and Novelty Risk

| Paper                                         | Year / Venue | Code                                | Mechanism                                                                           | Why Close                                                | Difference from New MAIN METHOD                                                                          | Novelty Risk                                   | Required Differentiation Experiment                                                                    |
| --------------------------------------------- | ------------ | ----------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| DeepSeekMath / GRPO                           | 2024, arXiv  | likely released around DeepSeekMath | Introduces GRPO as PPO variant for math reasoning and memory efficiency             | Base algorithmic family                                  | TRACE targets small-G sparse binary replay calibration, not GRPO itself                                  | Low as baseline, high if claiming GRPO novelty | Compare against stock GRPO/TRL under same env.([arXiv][26])                                            |
| Dr.GRPO / Understanding R1-Zero-like Training | 2025, arXiv  | paper/code likely                   | Identifies response-length bias in GRPO and proposes Dr.GRPO                        | Repo already adopts no per-row length normalization idea | TRACE adds replay trust/frontier; must not claim length-bias fix as own                                  | Medium                                         | Include Dr.GRPO baseline and length/reward diagnostics.([arXiv][27])                                   |
| DAPO                                          | 2025, arXiv  | open-source large-scale RL          | Decoupled Clip, Dynamic Sampling, large-scale RL improvements                       | Dynamic sampling/control close to prompt scheduling      | TRACE’s novelty must be verified replay trust and prompt posterior, not generic dynamic sampling         | Medium-high                                    | Reproduce DAPO or include official numbers only as secondary if compute prevents.([arXiv][28])         |
| RePO: Replay-Enhanced Policy Optimization     | 2025, arXiv  | likely                              | Replay buffer and diverse replay strategies for LLM RL                              | Closest mechanism: replay in policy optimization         | TRACE must differ via verified binary reward, prompt posterior trust, adaptive λ, staleness/drift budget | **High**                                       | Compare to RePO official; ablate trust gate vs generic replay buffer.([arXiv][25])                     |
| GSPO                                          | 2025, arXiv  | likely                              | Sequence-level importance ratio/clipping/rewarding for stability                    | Stability mechanism for RL LLM training                  | TRACE is replay-credit calibration; GSPO can be baseline or orthogonal combination                       | Medium                                         | Compare sequence-level GSPO vs TRACE on same G=2 binary setup.([arXiv][29])                            |
| “It Takes Two: Your GRPO Is Secretly DPO”     | 2025, arXiv  | unknown                             | Shows 2-GRPO can match larger group sizes; GRPO resembles DPO-like contrastive loss | Directly relevant to G=2 route                           | TRACE is not merely G=2; it adds calibrated positive memory under sparse verifier                        | Medium                                         | Include G=2 GRPO / DPO-like baseline; show replay trust adds beyond pairwise effect.([OpenReview][30]) |
| RLVR / CoT-Pass@K analysis                    | 2025, arXiv  | unknown                             | Questions whether RLVR improves reasoning or sampling efficiency                    | Evaluation mechanism risk                                | TRACE must measure reasoning/generalization, not only greedy answer on prefix                            | Medium                                         | Add pass@k / solution-quality / stratified difficulty analysis.([arXiv][31])                           |

**Novelty risk verdict:**
Naively saying “we add verified replay CE to GRPO” is likely **not novel enough**, especially after RePO. The only defensible novelty path is: **small-G binary GRPO + prompt-conditioned posterior trust + replay drift control + mechanism ablations proving generic replay is insufficient**.

---

# 第十三部分：Keep / Delete / Rewrite / Archive Plan

| Item                           | Type       | File / Directory / Claim / Experiment                        | Current Role            | Problem Under New MAIN PATH                           | Action                                    | Reason                                                    |
| ------------------------------ | ---------- | ------------------------------------------------------------ | ----------------------- | ----------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| README current focus           | Claim/doc  | `README.md`                                                  | Current focus           | Must reflect provisional TRACE route after validation | REWRITE                                   | Remove over-strong stable claim; keep retraction pointers |
| SPO+Replay proposal            | Doc        | `PROPOSAL_SPO_REPLAY.md`                                     | Current proposal        | Useful but now superseded by TRACE diagnosis          | ARCHIVE / REWRITE                         | Keep as evidence fragment, not final method               |
| Retractions                    | Integrity  | `RETRACTIONS.md`                                             | Retraction source       | None                                                  | FREEZE                                    | Do not erase negative/confounded history                  |
| Auto review                    | Integrity  | `review-stage/AUTO_REVIEW.md`                                | Reviewer risk           | None                                                  | FREEZE                                    | Preserve fatal weaknesses                                 |
| Paper draft                    | Paper      | `paper/main.tex`                                             | Stale CSD/rho paper     | Contradicts current project                           | REWRITE / ARCHIVE                         | Cannot submit old claims                                  |
| Main trainer                   | Code       | `src/aser_trainer_v14.py`                                    | Current ASER            | Has fixed replay and suspected mask/decode issues     | REWRITE or MERGE INTO NEW METHOD          | Use as scaffold for TRACE trainer                         |
| Prompt stats                   | Code       | `src/prompt_stats.py`                                        | EMA baseline/hardness   | Lacks posterior/trust/uncertainty                     | REWRITE                                   | New `PromptCreditState`                                   |
| Replay bank                    | Code       | `src/replay_bank.py`                                         | Uniform verified replay | Core missing mechanism                                | REWRITE                                   | New `TrustGatedReplayBank`                                |
| ASER launcher                  | Script     | `scripts/run_aser_mvp.py`                                    | Main training           | Needed as baseline                                    | FREEZE + KEEP ONLY AS BASELINE            | Do not mutate old results; fork new launcher              |
| Eval script                    | Script     | `scripts/eval_stratified.py`                                 | Eval                    | first-N issue                                         | REWRITE                                   | Add full/random/stratified and provenance                 |
| Full-set launcher              | Script     | `launch_fullset_eval.sh`                                     | Eval plan               | Outputs missing                                       | KEEP / REWRITE                            | Use for full benchmark gate                               |
| Analysis scripts               | Script     | `scripts/analyze_wave11.py`                                  | Result aggregation      | Provenance/label mismatch                             | REWRITE                                   | Enforce manifest schema                                   |
| `configs/aser_mvp.yaml`        | Config     | current config                                               | Baseline                | Not TRACE                                             | KEEP ONLY AS BASELINE                     | Add `trace_grpo_minimal.yaml`                             |
| phase α/β configs              | Config/exp | phase diagram                                                | Old scalar control      | Mechanistically weak                                  | ARCHIVE                                   | Keep negative evidence                                    |
| SPO+Replay n=200 results       | Results    | `analysis_wave11.json`                                       | Positive signal         | prefix/high variance                                  | KEEP ONLY AS HISTORICAL POSITIVE EVIDENCE | Must label reliability                                    |
| 500-step full-set              | Results    | `wave14_500step`                                             | Negative signal         | None                                                  | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE | Central motivation                                        |
| phase diagram results          | Results    | `wave14_phase_diagram`                                       | Negative signal         | None                                                  | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE | Rules out scalar route                                    |
| SFT-gold                       | Baseline   | wave13                                                       | Strong baseline         | Not main method                                       | KEEP ONLY AS BASELINE                     | Required comparator                                       |
| old rho/CSD/AdaBalance modules | Code/claim | `src/rho_*`, `csd_logging.py`, `adabalance.py`, paper claims | Old route               | Retracted/confounded                                  | ARCHIVE / KEEP ONLY AS BASELINE if needed | Prevent narrative contamination                           |
| true-dup/adaptive duplication  | Ablation   | run flags/results                                            | Scheduling route        | Unstable/no effect                                    | KEEP ONLY AS ABLATION                     | Not main                                                  |

---

# 第十四部分：Claude Code Implementation Plan

## Task 1: Freeze and label old routes

**Purpose:** 防止旧 rho/CSD/phase/dup 路线继续污染主线。
**Which Phenomenon / Constraint It Addresses:** P11, D9, D10.
**Why It Supports New MAIN METHOD PATH:** TRACE 必须从旧正负结果中学习，但不能继承旧 claim。
**Files to Inspect:** `README.md`, `PROPOSAL_SPO_REPLAY.md`, `RETRACTIONS.md`, `paper/main.tex`, `src/*rho*`, `src/csd_logging.py`, `src/adabalance.py`.
**Files to Edit:** `README.md`, new `docs/ARCHIVED_ROUTES.md`.
**Files to Delete / Archive:** none silently; move only if paths updated and imports unaffected.
**Functions / Classes:** none.
**Exact Change:** Add an “Archived / Not Mainline” section listing rho/CSD/AdaBalance/phase/dup as historical or ablation-only; link RETRACTIONS.
**Do Not Change:** Do not delete raw results or retractions.
**Verification Command:** `python -m compileall src scripts` and `grep -R "AdaBalance\\|CSD\\|rho" README.md paper docs -n`.
**Expected Result:** Old claims are visibly labeled stale/retracted.
**Failure Means:** Documentation still encourages wrong main route.
**Rollback Condition:** If imports break or archived files are needed by baseline scripts.
**Priority:** P0.
**Confidence:** high.

## Task 2: Add run/result provenance schema

**Purpose:** Make future results auditable.
**Which Phenomenon / Constraint It Addresses:** P12, R4 inconsistency, D7.
**Why It Supports New MAIN METHOD PATH:** TRACE gains only count if seeds/configs/checkpoints/eval ids are traceable.
**Files to Inspect:** `scripts/run_aser_mvp.py`, `scripts/analyze_wave11.py`, eval scripts.
**Files to Edit:** new `src/provenance.py`, update launch/eval scripts minimally.
**Exact Change:** Save JSON manifest with command, git hash, resolved config, seed, dataset split, eval ids, checkpoint path, adapter hash, package versions.
**Do Not Change:** Do not alter metrics or generation defaults.
**Verification Command:** `python scripts/run_aser_mvp.py --help`; add `python scripts/check_manifest_schema.py --manifest <sample>`.
**Expected Result:** Every new run writes a machine-readable manifest.
**Failure Means:** Future comparisons remain unreliable.
**Rollback Condition:** Manifest writing crashes training.
**Priority:** P0.
**Confidence:** high.

## Task 3: Fix evaluation protocol

**Purpose:** Remove first-N prefix as default paper evidence.
**Which Phenomenon / Constraint It Addresses:** P12, D7.
**Why It Supports New MAIN METHOD PATH:** TRACE must be evaluated on full or pre-registered subsets.
**Files to Inspect:** `scripts/eval_stratified.py`, `launch_fullset_eval.sh`.
**Files to Edit:** `scripts/eval_stratified.py`, new `scripts/check_eval_split.py`.
**Exact Change:** Add `--selection {first_n,full,random,stratified}`; require `--selection first_n` explicitly for legacy reproduction; default paper mode `full`; save question ids; compute accuracy denominator from actual evaluated length.
**Do Not Change:** Do not change answer extraction unless separately tested.
**Verification Command:**
`python scripts/eval_stratified.py --help`
`python scripts/check_eval_split.py --selection full --expected-n 1319`
**Expected Result:** Eval ids and selection method saved in every result JSON.
**Failure Means:** Positive results can still be prefix artifacts.
**Rollback Condition:** Full-set eval cannot load dataset.
**Priority:** P0.
**Confidence:** high.

## Task 4: Fix trainer mask and EOS decoding issues

**Purpose:** Remove possible silent training/eval contamination.
**Which Phenomenon / Constraint It Addresses:** code audit P0/P1.
**Why It Supports New MAIN METHOD PATH:** TRACE comparisons must not be polluted by padding or post-EOS tokens.
**Files to Inspect:** `src/aser_trainer_v14.py`.
**Files to Edit:** `src/aser_trainer_v14.py` or new fork `src/trace_grpo_trainer.py`.
**Exact Change:** Pass `attention_mask` to all model forward calls used for logprobs; decode completions only up to EOS / valid mask for reward.
**Do Not Change:** Do not change reward function semantics or answer regex without a separate metric test.
**Verification Command:** `pytest tests/test_attention_mask_logprobs.py tests/test_eos_reward_decode.py`.
**Expected Result:** Tests show mask affects padded inputs and EOS junk no longer changes reward.
**Failure Means:** Existing comparisons may remain contaminated.
**Rollback Condition:** TRL/model API rejects attention_mask path; then implement compatibility wrapper.
**Priority:** P0.
**Confidence:** medium-high.

## Task 5: Implement PromptCreditState

**Purpose:** Replace naive EMA-only stats with posterior evidence state.
**Which Phenomenon / Constraint It Addresses:** P7, P8, P9, D5.
**Why It Supports New MAIN METHOD PATH:** This is half of TRACE’s missing mechanism.
**Files to Inspect:** `src/prompt_stats.py`.
**Files to Edit:** new `src/prompt_credit_state.py`; keep old file for baseline.
**Functions / Classes:** `PromptCreditState`, `PromptCreditStore`.
**Exact Change:** Track `success_count`, `fail_count`, `alpha`, `beta`, `p_hat`, `uncertainty`, `frontier`, `baseline_ema`, `replay_exposure`, `last_seen_step`.
**Do Not Change:** Do not alter old `PromptStatsStore` used by legacy baseline.
**Verification Command:** `pytest tests/test_prompt_credit_state.py`.
**Expected Result:** Frontier high for evidence-backed uncertain prompts, low for unseen/easy/hopeless prompts.
**Failure Means:** TRACE gate cannot be trusted.
**Rollback Condition:** State update changes legacy ASER behavior.
**Priority:** P0.
**Confidence:** medium-high.

## Task 6: Implement TrustGatedReplayBank

**Purpose:** Replace uniform replay with trust-weighted replay.
**Which Phenomenon / Constraint It Addresses:** P2, P7, D3.
**Why It Supports New MAIN METHOD PATH:** This is TRACE’s central mechanism.
**Files to Inspect:** `src/replay_bank.py`.
**Files to Edit:** new `src/trust_gated_replay_bank.py`.
**Exact Change:** Store item metadata: prompt_id, token ids/text, reward, source_step, length, hash, replay_count, last_replayed_step; implement weighted sampling by `frontier * trust * age_decay * diversity * length_guard`.
**Do Not Change:** Do not delete old `ReplayBank`; keep for A baseline.
**Verification Command:** `pytest tests/test_trust_gated_replay_bank.py`.
**Expected Result:** Stale/overexposed/too-long items get lower probability; verified frontier items higher.
**Failure Means:** TRACE degenerates into fixed replay.
**Rollback Condition:** Sampling becomes empty too often; fall back to no replay with warning.
**Priority:** P0.
**Confidence:** medium.

## Task 7: Add TRACE-GRPO trainer

**Purpose:** Minimal implementation of new method.
**Which Phenomenon / Constraint It Addresses:** D1–D6.
**Why It Supports New MAIN METHOD PATH:** Implements adaptive λ_eff instead of fixed λ.
**Files to Inspect:** `src/aser_trainer_v14.py`.
**Files to Edit:** new `src/trace_grpo_trainer.py`.
**Exact Change:** Fork minimally from ASER; replace prompt stats/replay bank; compute `lambda_eff`; add budget penalty and logging; keep old ASER unchanged.
**Do Not Change:** Do not alter model, dataset, reward, optimizer, LoRA defaults.
**Verification Command:** `python scripts/run_trace_grpo.py --config configs/trace_grpo_minimal.yaml --seed 42 --max-steps 2 --output-dir results/smoke_trace`.
**Expected Result:** Runs two steps and logs λ_eff/frontier/trust metrics.
**Failure Means:** New mechanism not operational.
**Rollback Condition:** Trainer cannot run two steps.
**Priority:** P0.
**Confidence:** medium.

## Task 8: Add TRACE config and launcher

**Purpose:** Make A/B/C experiments reproducible.
**Which Phenomenon / Constraint It Addresses:** D10.
**Files to Inspect:** `scripts/run_aser_mvp.py`, `configs/aser_mvp.yaml`.
**Files to Edit:** `scripts/run_trace_grpo.py`, `configs/trace_grpo_minimal.yaml`.
**Exact Change:** Add switches: `--trace-mode full|constant_gate|no_replay`, `--lambda-max`, `--frontier-min-count`, `--age-tau`, `--replay-token-budget`.
**Do Not Change:** Legacy ASER command semantics.
**Verification Command:** `python scripts/run_trace_grpo.py --help`.
**Expected Result:** A/B/C modes are explicit and logged.
**Failure Means:** Ablations cannot isolate new mechanism.
**Rollback Condition:** CLI conflicts with existing scripts.
**Priority:** P0.
**Confidence:** high.

## Task 9: Add sanity tests

**Purpose:** Catch silent failures before expensive runs.
**Which Phenomenon / Constraint It Addresses:** implementation reliability risk.
**Files to Edit:** `tests/test_data_split.py`, `tests/test_metric_exact.py`, `tests/test_checkpoint_loading.py`, `tests/test_one_batch_overfit.py`.
**Exact Change:** Add minimal tests for split disjointness, answer extraction, checkpoint adapter load, and one-batch overfit.
**Verification Command:** `pytest tests -q`.
**Expected Result:** All local sanity tests pass.
**Failure Means:** Do not run benchmark.
**Rollback Condition:** Tests are flaky due external dataset download; mock dataset locally.
**Priority:** P0.
**Confidence:** high.

## Task 10: Run minimal A/B/C verification

**Purpose:** Prove new method is not just old positive fragment.
**Which Phenomenon / Constraint It Addresses:** D10.
**Files to Edit:** no code except experiment configs.
**Exact Change:** Run seeds 42/43/44 for: A fixed SPO+Replay, B TRACE constant gate, C full TRACE; evaluate n=200 legacy and full 1319.
**Do Not Change:** Dataset, metric, generation args, model, LoRA, max steps.
**Verification Command:**
`bash scripts/run_minimal_trace_abc.sh`
`python scripts/eval_stratified.py --selection full --n 1319 ...`
`python scripts/analyze_trace_abc.py`
**Expected Result:** C beats A and B on full-set mean with lower or equal variance and no 500-step collapse.
**Failure Means:** TRACE mechanism unsupported; stop or pivot.
**Rollback Condition:** Any A/B/C condition not comparable.
**Priority:** P0.
**Confidence:** medium-low.

## Task 11: Mechanism ablations

**Purpose:** Identify which TRACE sub-mechanism matters.
**Files to Edit:** configs only.
**Exact Change:** Run `no_age_decay`, `no_frontier`, `no_budget`, `no_length_guard`.
**Verification Command:** `bash scripts/run_trace_mechanism_ablation.sh`.
**Expected Result:** Removing core gate components degrades full-set or stability.
**Failure Means:** Claimed mechanism not supported.
**Priority:** P1.
**Confidence:** medium.

## Task 12: Rewrite paper only after experiments pass

**Purpose:** Prevent premature claim inflation.
**Files to Edit:** `paper/main.tex`, `README.md`.
**Exact Change:** Replace CSD/rho thesis with TRACE thesis only if minimal verification passes; include negative history transparently.
**Verification Command:** `grep -R "AdaBalance\\|CSD\\|rho control" paper/main.tex README.md -n`.
**Expected Result:** No stale main claims remain.
**Failure Means:** Paper still violates internal evidence.
**Priority:** P2.
**Confidence:** high.

---

# 第十五部分：Minimal Verification Experiments

| Priority | Experiment                               | Hypothesis                                   | Command                                                                                             | Config           | Dataset                 | Seeds        | Metric          | Success Criterion                                 | Failure Interpretation           |
| -------: | ---------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------- | ----------------------- | ------------ | --------------- | ------------------------------------------------- | -------------------------------- |
|        0 | Smoke test                               | New trainer runs                             | `python scripts/run_trace_grpo.py --config configs/trace_grpo_minimal.yaml --seed 42 --max-steps 2` | TRACE minimal    | GSM8K train tiny        | 42           | no crash + logs | Saves manifest and λ_eff logs                     | Implementation broken            |
|        0 | Data sanity check                        | Train/test disjoint and eval ids saved       | `python scripts/check_eval_split.py --selection full --expected-n 1319`                             | eval             | GSM8K                   | NA           | split ids       | No overlap, ids saved                             | Data leakage/provenance bug      |
|        0 | Metric sanity check                      | Answer extraction correct                    | `pytest tests/test_metric_exact.py`                                                                 | test             | synthetic GSM8K strings | NA           | exact match     | Known strings pass                                | Metric unreliable                |
|        0 | One-batch overfit                        | Trainer can learn trivial batch              | `python scripts/run_trace_grpo.py --one-batch-overfit --max-steps 50`                               | tiny             | 1–4 prompts             | 42           | train reward/CE | Overfits toy                                      | Gradient/loss bug                |
|        0 | Checkpoint loading check                 | Eval loads intended adapter                  | `python scripts/check_checkpoint_loading.py --adapter <path>`                                       | NA               | NA                      | NA           | hash match      | Loaded adapter hash matches manifest              | Stale checkpoint risk            |
|        1 | Reproduce current negative               | 500-step fixed SPO+Replay collapses full-set | existing eval on wave14 checkpoints                                                                 | legacy           | GSM8K full              | 42/43/44     | acc             | Reproduces ~44–45 mean                            | If not, provenance mismatch      |
|        1 | Reproduce current best positive fragment | Existing SPO+Replay n=200 signal exists      | `scripts/run_aser_mvp.py ...` then first_n eval                                                     | `aser_mvp.yaml`  | GSM8K first 200 + full  | 42/43/44     | acc             | n=200 comparable; full reported too               | Cannot compare TRACE to old best |
|        1 | Mechanism activation check               | TRACE actually gates replay                  | `python scripts/run_trace_grpo.py --trace-mode full --max-steps 50`                                 | TRACE            | train                   | 42           | λ_eff stats     | λ_eff nonconstant, frontier-dependent             | New method inactive              |
|        1 | New MAIN METHOD minimal test             | TRACE improves full-set vs fixed             | `bash scripts/run_minimal_trace_abc.sh --steps 200`                                                 | A/B/C            | GSM8K full              | 42/43/44     | acc             | C > A and B by ≥3 pp mean                         | Mechanism unsupported            |
|        1 | Key ablation: remove new mechanism       | Constant gate should lose benefit            | `--trace-mode constant_gate`                                                                        | B                | full                    | 42/43/44     | acc             | B ≈ A or < C                                      | If B=C, trust gate unnecessary   |
|        1 | A. Existing Best Positive Fragment Only  | Old positive fragment baseline               | `run_aser_mvp.py --backbone spo --lambda-rep 0.05`                                                  | A                | first200/full           | 42/43/44     | acc             | Establish baseline                                | If A dominates, new route weak   |
|        1 | B. New MAIN METHOD Without New Mechanism | New infra alone not enough                   | `run_trace_grpo.py --trace-mode constant_gate`                                                      | B                | first200/full           | 42/43/44     | acc             | B not > C                                         | If B≈C, novelty fails            |
|        1 | C. Full New MAIN METHOD                  | Trust gate causes gain                       | `run_trace_grpo.py --trace-mode full`                                                               | C                | first200/full           | 42/43/44     | acc/std         | C best full-set, lower variance                   | If fail, stop/pivot              |
|        2 | Small baseline comparison                | TRACE beats weak internal baselines          | run SPO-only/RFT/fixed-rho eval                                                                     | baseline configs | full                    | 42/43/44     | acc             | C > SPO/RFT/fixed-rho                             | If not, method fails             |
|        2 | Multi-seed stability                     | Effect not best-seed                         | run A/B/C seeds 42–46                                                                               | same             | full                    | 5 seeds      | mean/std/CI     | C improves mean, not just best seed               | Cherry-pick risk                 |
|        2 | Expansion gate to full benchmark         | Full-set gain survives all 1319              | `eval_stratified.py --selection full`                                                               | same             | GSM8K full              | all seeds    | acc             | Full result positive                              | Prefix artifact                  |
|        2 | Official baseline reproduction           | Mechanism-level comparison fair              | run official RePO/DAPO/Dr.GRPO if feasible                                                          | official         | GSM8K                   | ≥3           | acc             | Comparable or honest weaker claim                 | Novelty/baseline risk            |
|        2 | Unified environment comparison           | No env mismatch                              | `python scripts/compare_manifests.py`                                                               | all              | all                     | all          | env hash        | Same env/package versions                         | Comparison invalid               |
|        3 | Robustness/generalization                | Method not GSM8K-prefix-specific             | port to MATH subset or GSM8K random stratified                                                      | TRACE            | second dataset/subset   | 3            | acc/pass@k      | Same direction                                    | Dataset overfit                  |
|        3 | Statistical significance / CI            | Gain reliable                                | `python scripts/bootstrap_ci.py results/trace_abc/*.json`                                           | results          | full                    | ≥5 preferred | CI/p-value      | CI excludes zero or effect practically meaningful | Insufficient evidence            |

---

# 第十六部分：Baseline and SOTA Plan

| Baseline                   | Why Required                                   |            Official Code | Dataset                              | Metric       | Reproduction Requirement                        | Fairness Risk                                               |
| -------------------------- | ---------------------------------------------- | -----------------------: | ------------------------------------ | ------------ | ----------------------------------------------- | ----------------------------------------------------------- |
| Base Qwen/Qwen3.5-9B       | Establish model starting point                 |           model official | GSM8K full                           | greedy acc   | Same prompt/eval                                | Prompt formatting differences                               |
| Stock TRL GRPO             | Simplest algorithmic baseline                  |             TRL official | GSM8K full                           | acc          | Same G, model, LoRA, steps                      | TRL version differences                                     |
| Dr.GRPO                    | Length-bias/stability baseline                 | paper/TRL implementation | GSM8K full                           | acc + length | Same normalization choices                      | Accidentally using weaker config                            |
| SPO-only                   | Isolates per-prompt baseline                   |                     repo | GSM8K full                           | acc          | Same code/env                                   | Only n=3 not enough                                         |
| Existing fixed SPO+Replay  | A control; old positive fragment               |                     repo | first200/full                        | acc          | Same seeds/checkpoints                          | Must not tune TRACE more                                    |
| RFT-only                   | Tests replay CE alone                          |                     repo | full                                 | acc          | Fix provenance label issue                      | If poorly implemented, unfair                               |
| fixed-rho 0.70             | Old internal baseline                          |                     repo | full                                 | acc          | Same env, seeds                                 | Retracted/confounded old code                               |
| SFT-gold                   | Strong simple baseline / upper bound           | repo script if available | full                                 | acc          | Same data budget disclosed                      | Uses supervised gold traces, may not be same setting        |
| RePO                       | Closest replay mechanism                       |    official if available | GSM8K/full and maybe math benchmarks | acc          | Official code/config or faithful implementation | High novelty risk if omitted                                |
| DAPO                       | Dynamic sampling/control mechanism             |     official open-source | math RL benchmark                    | acc          | Official config when compute allows             | Scale mismatch                                              |
| GSPO                       | Sequence-level stable policy opt               |    official if available | GSM8K/math                           | acc          | Same backbone/steps if possible                 | Different objective may need tuning                         |
| 2-GRPO / DPO-like baseline | Tests whether G=2 contrastive effect is enough |  paper code if available | GSM8K                                | acc          | Same G=2                                        | If omitted, reviewer may say TRACE is pairwise DPO artifact |

---

# 第十七部分：Paper Thesis Reconstruction

1. **New Paper Thesis:**
   Sparse binary RLVR at small group size fails not only because group-relative advantages degenerate, but because rare successful trajectories are reused without calibrated trust. Prompt-conditioned trust-gated replay turns rare successes into stable learning signals.

2. **Main Technical Contribution:**
   A posterior-based prompt credit state and trust-gated replay objective for small-G binary GRPO.

3. **Main Empirical Claim:**
   Only if experiments pass: TRACE-GRPO improves full GSM8K accuracy and stability over fixed SPO+Replay, SPO-only, RFT-only, stock GRPO/Dr.GRPO, and generic replay baselines under identical evaluation.

4. **What Previous Failures Taught Us:**
   Fixed scalar control, uniform replay, hard duplication, and prefix-only evaluation can produce misleading positives.

5. **What We Should Not Claim:**
   Do not claim SOTA, universal RLVR improvement, solved math reasoning, or novelty of replay itself. Do not claim old rho/CSD/AdaBalance contribution.

6. **What We Can Claim If Experiments Pass:**
   Trust calibration, not replay alone, is the key stabilizer for verified replay in small-G binary GRPO.

7. **Required Baselines:**
   A/B/C controls, SFT-gold, Dr.GRPO, RePO, DAPO/GSPO if feasible, stock GRPO.

8. **Required Ablations:**
   no trust gate, no frontier, no age decay, no budget, no length guard, SPO-only, replay-only.

9. **Required Robustness Tests:**
   full GSM8K, random/stratified subsets, second math dataset or MATH subset, multi-seed, longer-horizon 500-step.

10. **Reviewer Likely Objections:**
    “This is just RePO”; “only GSM8K”; “first-200 cherry-pick”; “baselines weak”; “mechanism not proven”; “negative results hidden”.

11. **How New MAIN METHOD Answers Them:**
    RePO comparison, full-set eval, pre-registered A/B/C, negative evidence retained, mechanism logging.

12. **What Would Make This NeurIPS-Strong:**
    Full-set multi-seed gains, mechanism ablations, fair official baselines, transparent retractions, at least one additional dataset.

13. **What Would Make This Rejected:**
    Only n=200 positives, no RePO/DAPO/Dr.GRPO baseline, no mechanism ablation, paper still claiming CSD/rho.

14. **What Would Be Required for Oral-Level Strength:**
    Broad benchmark suite, strong effect over official baselines, clear theory or diagnostic explaining trust-calibrated replay, robust scaling.

15. **What Would Be Required for Best-Paper-Level Strength:**
    A general principle for sparse-verifier RL with rigorous analysis, broad reproducibility, strong practical adoption, and convincing evidence that the mechanism changes how RLVR is understood.

---

# 第十八部分：Reviewer Risk Assessment

| Risk                       | Why Reviewer May Object                               | Evidence Needed                           | How New MAIN METHOD Addresses It                     | Remaining Weakness                |
| -------------------------- | ----------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| Novelty risk               | RePO already does replay-enhanced policy optimization | Direct RePO comparison                    | TRACE differentiates by prompt posterior/trust/drift | Still may look incremental        |
| Incremental risk           | Could be fixed replay + heuristics                    | A/B/C and component ablations             | Show trust gate, not replay, causes gain             | Mechanism must be clean           |
| Baseline weakness          | Current baselines are internal/weak                   | Official Dr.GRPO/RePO/DAPO/GSPO           | Baseline plan includes official code                 | Compute burden                    |
| Reproducibility            | Provenance inconsistent                               | manifests, commands, configs, checkpoints | Add provenance schema                                | Old results remain messy          |
| Cherry-picking             | n=200 prefix positive                                 | full-set and eval ids                     | Fix eval protocol                                    | Need upload full logs             |
| Negative result hiding     | Many failed routes                                    | Keep retractions and negative tables      | Use failures as motivation                           | Paper space pressure              |
| Overclaiming               | Stale paper claims too strong                         | rewrite paper                             | Remove CSD/rho/SOTA claims                           | Need discipline                   |
| Unclear mechanism          | Trust gate may be ad hoc                              | logs + ablations                          | λ_eff/frontier/trust diagnostics                     | Theory may be thin                |
| Ablation insufficiency     | Need prove not old fragment                           | A/B/C mandatory                           | Directly tests old vs new                            | More components require more runs |
| Dataset limitation         | GSM8K only                                            | second dataset                            | Add MATH/subset                                      | Compute/time                      |
| Compute unfairness         | DAPO/RePO may use different scale                     | same model/steps or disclose              | unified environment                                  | May not match official scale      |
| Implementation reliability | mask/EOS bugs possible                                | tests before runs                         | fix P0/P1 bugs                                       | Results before fix downgraded     |
| Related work omission      | Fast-moving RLVR literature                           | current related work review               | include closest mechanisms                           | Needs continual update            |

---

# 第十九部分：Final Decision

## 1. One-Sentence Verdict

当前项目应停止把 fixed SPO+Replay n=200 正面结果包装为主方法，转向 **TRACE-GRPO：prompt-conditioned trust-calibrated replay credit with drift control**，并用 A/B/C full-set 多 seed 实验证明新增机制而不是旧正面片段带来收益。

## 2. Current Most Likely Root Cause

最可能根因是组合型：

* **missing mechanism:** fixed verified replay 缺少 prompt-conditioned trust / uncertainty / drift control；
* **evaluation issue:** first-200 deterministic prefix 使正面结果不够可信；
* **implementation risk:** attention_mask / EOS decoding / provenance 可能污染结果；
* **baseline mismatch:** SFT-gold 与机制近邻 official baselines 尚未公平覆盖；
* **method assumption failure:** “verified successes can be uniformly replayed”这个假设不成立。

## 3. Why This Is Not Just the Existing Best Path

Existing best path 是 `SPO + fixed λ verified replay CE`。TRACE-GRPO 的主变量是 `λ_eff(x,y,t)`，由 prompt posterior、frontier、trust、age decay、replay exposure 和 drift budget 共同决定。它能退化成 existing best path，因此可以被直接 ablate；这不是换名或调参。

## 4. Phenomena Explained

* first-200 正面：rare successes + SPO baseline 有效；
* 500-step collapse：uniform fixed replay 造成 drift / stale overfit；
* phase null：global scalar α/β 不是关键；
* RFT-only weak：CE alone 不够；
* SPO-only limited：baseline alone 不够；
* true-dup unstable：hardness without uncertainty 不可靠；
* SFT-gold gap：需要强 baseline 与更严谨 claim。

## 5. Mechanism Missing in Current Method

缺少 **“成功轨迹何时可信、对哪个 prompt 有用、该 replay 多强、何时停止”** 的结构化机制。

## 6. New Mechanism

**Prompt posterior + trust-gated replay + adaptive λ_eff + drift budget**。

## 7. What to Delete / Archive / Rewrite

* **Archive / freeze:** rho/CSD/AdaBalance route, phase α/β route, adaptive duplication as main route。
* **Rewrite:** `paper/main.tex`, `README.md`, `eval_stratified.py`, result aggregation。
* **Rewrite into new modules:** `prompt_stats.py → prompt_credit_state.py`, `replay_bank.py → trust_gated_replay_bank.py`, `aser_trainer_v14.py → trace_grpo_trainer.py`。
* **Keep as baseline/ablation:** fixed SPO+Replay, SPO-only, RFT-only, SFT-gold, fixed-rho, true-dup。

## 8. First Five Claude Code Tasks

1. Freeze old routes and label retracted/stale claims.
2. Add provenance schema for every new run/eval.
3. Fix eval protocol: full-set/default, explicit first-N legacy mode, save eval ids.
4. Fix trainer attention_mask and EOS decoding issues.
5. Implement `PromptCreditState` and tests.

## 9. Minimal Experiments

* Smoke / data / metric / checkpoint / one-batch overfit.
* Reproduce current fixed SPO+Replay positive and 500-step negative.
* A/B/C:

  * A: Existing Best Positive Fragment Only.
  * B: TRACE infrastructure without trust gate.
  * C: Full TRACE-GRPO.
* Full GSM8K n=1319, seeds 42/43/44 minimum, then 5 seeds if positive.
* Mechanism ablations: no frontier, no trust age decay, no budget.

## 10. Continue / Stop / Pivot Criteria

**Continue** if C beats A and B on full-set mean by practical margin, variance not worse, and 500-step no collapse.
**Stop** if C only improves first-200 but not full-set.
**Pivot** if B ≈ C, meaning infrastructure or tuning explains gains, not trust mechanism.
**Hard stop** if official RePO / Dr.GRPO / DAPO baseline dominates and TRACE has no differentiating mechanism evidence.

## 11. NeurIPS-Level Gap

Need full-set multi-seed results, official mechanism-neighbor baselines, clean provenance, additional dataset, ablations proving trust gate, and rewritten paper.

## 12. Oral / Best Paper Gap

Oral-level requires broad benchmark and strong mechanism analysis. Best-paper-level would require a general, widely validated principle of sparse-verifier RL, not just GSM8K engineering.

## 13. Confidence

**medium-low.** The diagnosis is well supported by current positive/negative/unstable phenomena, but the proposed method is still an evidence-backed hypothesis requiring minimal falsification.

---

# 第二十部分：Final Claude Code Instruction

```text
Claude Code, execute the following plan.

You must implement the New MAIN METHOD PATH defined in the GPT-5.5 Pro diagnosis report:

TRACE-GRPO: Trust-Calibrated Replay and Prompt-Conditioned Credit Assignment for Sparse Binary GRPO.

Do not invent a different method.
Do not optimize for superficial positive results.
Do not weaken baselines.
Do not delete negative results silently.
Do not change metrics or datasets unless explicitly instructed.
Do not rewrite unrelated files.
Do not use first-200 GSM8K prefix as the main paper result.
Do not revive retracted rho/CSD/AdaBalance claims.

Your tasks are:

1. Freeze old routes.
   - Add documentation marking rho/CSD/AdaBalance/phase/dup routes as archived, retracted, baseline-only, or ablation-only.
   - Do not delete raw results.
   - Keep RETRACTIONS.md and review-stage notes intact.

2. Add provenance logging.
   - Create a manifest schema that records command, git hash, config hash, seed, model, dataset split, eval ids, checkpoint path, adapter hash, package versions, and generation args.
   - Every new train/eval run must write this manifest.

3. Fix evaluation.
   - Update scripts/eval_stratified.py to support explicit --selection {first_n,full,random,stratified}.
   - The main evaluation mode must be full GSM8K test.
   - Save exact evaluated question ids.
   - Compute accuracy using the actual evaluated count, not a requested n if fewer examples are loaded.
   - Preserve first_n only for legacy reproduction.

4. Fix trainer correctness issues before new experiments.
   - Pass attention_mask to model forward calls used for logprobs.
   - Decode completions for reward only up to EOS / valid completion mask.
   - Add tests for padding-mask logprob behavior and EOS reward decoding.
   - Do not change reward semantics beyond removing post-EOS contamination.

5. Implement PromptCreditState.
   - Add src/prompt_credit_state.py.
   - Track per-prompt success_count, fail_count, alpha, beta, p_hat, uncertainty, frontier score, baseline EMA, replay exposure, last_seen_step.
   - Add unit tests showing frontier is high only for evidence-backed uncertain prompts and low for unseen/easy/hopeless prompts.

6. Implement TrustGatedReplayBank.
   - Add src/trust_gated_replay_bank.py.
   - Store verified success metadata: prompt_id, completion tokens/text, reward, source_step, length, hash, replay_count, last_replayed_step.
   - Implement weighted sampling using prompt frontier, trust, age decay, diversity, length guard, and replay exposure.
   - Keep old ReplayBank unchanged for baseline A.

7. Implement TRACE-GRPO trainer.
   - Create src/trace_grpo_trainer.py by minimally forking the current ASER trainer.
   - Replace fixed lambda replay with lambda_eff(x,y,t).
   - lambda_eff must depend on PromptCreditState and TrustGatedReplayBank trust scores.
   - Add replay token budget / drift budget.
   - Log lambda_eff distribution, frontier histogram, replay age, replay token ratio, replay exposure, all-pass/all-fail rates, KL, length, reward, and loss components.
   - Do not change backbone, LoRA defaults, dataset, metric, or generation args.

8. Add TRACE launcher and configs.
   - Create scripts/run_trace_grpo.py.
   - Create configs/trace_grpo_minimal.yaml.
   - Add explicit modes:
     A. existing_best_fixed_spo_replay
     B. trace_constant_gate_without_new_mechanism
     C. full_trace_grpo
   - All modes must save manifests and comparable logs.

9. Add sanity checks.
   - Add tests/scripts for data split, metric extraction, checkpoint loading, one-batch overfit, and manifest validation.
   - Run these before any benchmark experiments.
   - Stop if any sanity check fails.

10. Run minimal verification experiments only after tests pass.
   - Reproduce current fixed SPO+Replay positive fragment.
   - Reproduce current 500-step full-set negative result if checkpoints are available.
   - Run A/B/C on GSM8K full test n=1319 with seeds 42, 43, 44:
       A. Existing Best Positive Fragment Only
       B. New MAIN METHOD Without New Mechanism
       C. Full New MAIN METHOD
   - Evaluate also on legacy first_n=200 only as a diagnostic, not as the main claim.
   - Save all logs, manifests, result JSONs, and eval ids.

11. Run mechanism ablations if and only if C beats A and B on full-set.
   - no_frontier
   - no_age_decay
   - no_drift_budget
   - no_length_guard
   - no_replay_exposure_penalty

12. Do not rewrite the paper until minimal experiments pass.
   - If experiments pass, rewrite paper/main.tex around TRACE-GRPO.
   - Remove CSD/rho/AdaBalance main claims.
   - Include negative results transparently.
   - Include A/B/C ablation and full-set multi-seed results.

For every task:
- make the smallest necessary change;
- show the diff;
- run the specified verification command;
- save logs;
- report failures;
- stop if verification fails;
- do not proceed to full benchmark until minimal tests pass.

At the end, output:
- files changed;
- files archived;
- configs added;
- commands run;
- logs;
- result table;
- failed checks;
- unresolved issues;
- whether Full New MAIN METHOD beats:
  A. Existing Best Positive Fragment Only,
  B. New MAIN METHOD Without New Mechanism,
  C. Full New MAIN METHOD.
```

[1]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/README.md "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/README.md"
[2]: https://github.com/Sunshine535/nips-grpo-dynamics "GitHub - Sunshine535/nips-grpo-dynamics: Phase diagrams and zero-score gradient reshaping for GRPO training stability · GitHub"
[3]: https://github.com/Sunshine535/nips-grpo-dynamics/tree/main/src "nips-grpo-dynamics/src at main · Sunshine535/nips-grpo-dynamics · GitHub"
[4]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/paper/main.tex "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/paper/main.tex"
[5]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/scripts/run_aser_mvp.py "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/scripts/run_aser_mvp.py"
[6]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/scripts/eval_stratified.py "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/scripts/eval_stratified.py"
[7]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/configs/aser_mvp.yaml "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/configs/aser_mvp.yaml"
[8]: https://github.com/Sunshine535/nips-grpo-dynamics/tree/main/results "nips-grpo-dynamics/results at main · Sunshine535/nips-grpo-dynamics · GitHub"
[9]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/review-stage/AUTO_REVIEW.md "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/review-stage/AUTO_REVIEW.md"
[10]: https://github.com/Sunshine535/nips-grpo-dynamics/raw/refs/heads/main/RETRACTIONS.md "raw.githubusercontent.com"
[11]: https://github.com/Sunshine535/nips-grpo-dynamics/raw/refs/heads/main/results/analysis_wave11.json "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/requirements.txt "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/requirements.txt"
[13]: https://github.com/Sunshine535/nips-grpo-dynamics/blob/main/PROPOSAL_SPO_REPLAY.md "nips-grpo-dynamics/PROPOSAL_SPO_REPLAY.md at main · Sunshine535/nips-grpo-dynamics · GitHub"
[14]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/src/aser_trainer_v14.py "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/src/aser_trainer_v14.py"
[15]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/src/prompt_stats.py "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/src/prompt_stats.py"
[16]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/src/replay_bank.py "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/src/replay_bank.py"
[17]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/launch_fullset_eval.sh "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/launch_fullset_eval.sh"
[18]: https://github.com/Sunshine535/nips-grpo-dynamics/raw/refs/heads/main/results/round2_analysis.json "raw.githubusercontent.com"
[19]: https://github.com/Sunshine535/nips-grpo-dynamics/raw/refs/heads/main/results/wave10_training_dynamics.json "raw.githubusercontent.com"
[20]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/wave14_500step/evals/eval_seed42.json "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/wave14_500step/evals/eval_seed42.json"
[21]: https://github.com/Sunshine535/nips-grpo-dynamics/tree/main/results/wave14_phase_diagram "https://github.com/Sunshine535/nips-grpo-dynamics/tree/main/results/wave14_phase_diagram"
[22]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/stratified_eval_wave13/sft_gold_seed42.json "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/stratified_eval_wave13/sft_gold_seed42.json"
[23]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/wave14_phase_diagram/evals/eval_a0.1_b0.0.json "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/wave14_phase_diagram/evals/eval_a0.1_b0.0.json"
[24]: https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/stratified_eval_wave13/truedup_seed42.json "https://raw.githubusercontent.com/Sunshine535/nips-grpo-dynamics/main/results/stratified_eval_wave13/truedup_seed42.json"
[25]: https://arxiv.org/abs/2506.09340 "https://arxiv.org/abs/2506.09340"
[26]: https://arxiv.org/abs/2402.03300 "https://arxiv.org/abs/2402.03300"
[27]: https://arxiv.org/abs/2503.20783 "https://arxiv.org/abs/2503.20783"
[28]: https://arxiv.org/abs/2503.14476 "https://arxiv.org/abs/2503.14476"
[29]: https://arxiv.org/abs/2507.18071 "https://arxiv.org/abs/2507.18071"
[30]: https://openreview.net/pdf?id=evlIZKNVt7 "https://openreview.net/pdf?id=evlIZKNVt7"
[31]: https://arxiv.org/abs/2506.14245 "https://arxiv.org/abs/2506.14245"
