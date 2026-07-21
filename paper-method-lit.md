# 方法侧文献：查重判决与借鉴清单（原始综合器输出）

> 生成于 2026-07-18 深夜；四巷道搜索（删失标签/决策聚焦/收缩估计/可实施alpha）+ 综合器实读验证。
> 完整巷道明细见 workflow 转录。

```json
{
"verification": [
{
"item": "Du 2025, arXiv:2507.07107 (C4 highest threat)",
"fetched": "arxiv.org/abs/2507.07107",
"confirmed": true,
"findings": "Title/author confirmed (Yimin Du). Label treatment confirmed as HARD DROP-MASK, not censored regression: 'a Boolean tradability mask is constructed at data load time and threaded through every operator, so that no window ever reads a non-tradable price'. Optimizer is Markowitz-Ledoit-Wolf via cvxpy — shrinkage exists but ONLY on the covariance matrix, none on alpha scores, no EB/posterior, no feasibility-weighted objective. Gap numbers confirmed: unmasked IC inflated 18%, mask contributes +0.44 Sharpe (largest single ablation contributor).",
"implication": "Du is the closest single composition: crude versions of all three slots (deletion labels + tradable-only optimizer input + covariance-only shrinkage). Our differentiation line is exact: deletion is not-at-random under signal-correlated censoring, so correction (using the observed indicator) strictly dominates deletion; and shrinkage must move to the alpha/posterior, not the covariance."
},
{
"item": "Grinold/Barra scale-and-trim (C3 highest threat)",
"fetched": "MSCI Barra 'Converting Scores into Alphas' (May 2010, Gleiser & McKenna) — full 13-page PDF text extracted",
"confirmed": true,
"findings": "Confirms alpha = IC x residual_volatility x score with cross-sectionally standardized scores and IC 0.05 'good' / 0.10 'very good' — i.e., built-in 10-20x shrinkage of raw z-scores toward zero; 'consensus forecasts imply no alphas and lead to benchmark holdings'. The explicit trim/cap rule is NOT in this note (it lives in Grinold-Kahn Active Portfolio Mgmt 2000 ch.14 refinement / Northfield 2007). ZERO mention of feasibility, price limits, or censoring anywhere in the document.",
"implication": "Threat stands (IC-shrinkage of scores is 30-year practice and must be the C3 baseline) but the defense is verified: the entire Grinold stack is censoring-blind — IC is estimated on realized (censored) returns and the scale is signal-extremity-independent. C3 novelty must be censoring-coupled shrinkage, never shrinkage per se."
}
],
"per_component_table": [
{
"component": "C1 (LABEL: tradable-member mean returns, hard mask)",
"closest_existing": "Du 2025 arXiv:2507.07107 (same market, same problem, deletion mask — VERIFIED); estimator layer: Grabit (Sigrist-Hirnschall, JBF 2019) Tobit-in-boosting; Deep Tobit (Neural Networks 2021); censored quantile NN (Pearce et al., NeurIPS 2022); Heckman-NN (arXiv 2309.08043); Hüttel censored demand QRNN (per-sample known thresholds)",
"borrow": "Two-sided Tobit type-I likelihood with per-stock-day KNOWN thresholds: uncensored samples contribute the Gaussian density of (y - f(x)); limit-up samples contribute the log survival probability that the latent return exceeds the known +limit given f(x) and sigma; mirrored for limit-down. Equivalent EM form: replace each censored label with its conditional expectation E[y*|y* beyond limit, x] (inverse-Mills-ratio formula), refit, iterate. Optionally add a Heckman-style supervised selection head (indicator is observed) with control-function term for next-day-open executability MNAR structure; heteroscedastic sigma head from Danaila-Buiu.",
"remaining_delta": "Correction instead of deletion where deletion is provably not-at-random (censoring iff y crosses the limit, correlated with signal extremes); observed per-sample censor indicator + known thresholds = strictly more informative regime than the entire survival/truncation literature assumes (Daskalakis et al. gives identifiability as a lower bound). Nobody does censoring-corrected label estimation for cross-sectional A-share alpha.",
"verdict": "Estimator novelty: DEAD. Application + MNAR framing + coupling to execution: SAFE. Threat: MEDIUM overall; Du must be cited and beaten as the deletion baseline."
},
{
"component": "C2 (DECISION: hold mid-band 60-90 of score distribution)",
"closest_existing": "BPQP (NeurIPS 2024, end-to-end MVO on A-shares — mandatory baseline); SPO+ (Elmachtoub-Grigas, Mgmt Sci 2022); JKMP implementable efficient frontier (RFS); Feasibility-aware DFL (arXiv 2510.04951); Novy-Marx-Velikov buy/hold hysteresis bands (RFS 2016); sleeping bandits (COLT 2008) for reward-correlated availability hardness; Yang-Zhang (JFM 2019) exclude-extremes momentum",
"borrow": "(a) Differentiable portfolio layer (cvxpylayers/BPQP backward) with per-name box constraints zeroed on the locked side by the OBSERVED censoring state; (b) SPO+/post-hoc-regret loss on executable return (loss = realized suboptimality of the decision induced by predicted scores over the feasible set, computed with one argmax oracle call); (c) feasibility shadow price from bandits-with-knapsacks: adjusted score = score - lambda x P(limit-hit|x), lambda learned; (d) 'optimism on reward, pessimism on feasibility' from stage-wise constrained bandits; (e) Liang 2025 features for the supervised P(limit-hit|x) head.",
"remaining_delta": "No existing work has a feasible set that is simultaneously (a) state-dependent, (b) OBSERVED at decision time, (c) correlated with reward extremes, and (d) the same mechanism censoring the training labels. JAIR 2024 DFL survey explicitly lists constraint-side parameters + uncertainty-aware DFL as open. Mid-band must be differentiated from Novy-Marx-Velikov cost-hysteresis by generating mechanism (censoring, not costs).",
"verdict": "Generic decision-focused portfolio slot: DEAD (BPQP/SPO own it). Observed signal-correlated feasible set + executable-return objective: SAFE/OPEN. Threat: MEDIUM; BPQP required as baseline."
},
{
"component": "C3 (ESTIMATION: 9-model equal-weight consensus)",
"closest_existing": "Grinold IC-scaling + Grinold-Kahn scale-and-trim + TE-constrained optimization (VERIFIED censoring-blind — the baseline to beat); Efron 2011 Tweedie formula (JASA); Harvey-Liu random-effects EB on alphas (RFS 2018); Jorion Bayes-Stein (JFQA 1986); Chen-Zimmermann closed-form shrinkage factor (RAPS 2020); AKM winner's-curse inference (QJE 2024); Mogstad et al. rank confidence sets",
"borrow": "(a) Tweedie posterior: corrected score = z + sigma^2 x slope of log empirical marginal density at z, estimated each rebalance date from the day's cross-section (Lindsey Poisson-spline) — shrinks extremes nonlinearly and data-adaptively, making 'avoid the extremes' endogenous; (b) hierarchical precision weighting: per-stock sigma^2 = cross-model disagreement variance of the 9 scores, so high-disagreement stocks shrink hardest (Harvey-Liu/Bayes-Stein weight = noise variance over noise-plus-signal variance); (c) AKM truncated-normal median-unbiased estimate for honest reporting of the selected basket's expected return.",
"remaining_delta": "Censoring-coupled shrinkage: at signal extremes the observed z is itself a censored observation (plain Gaussian-noise Tweedie/EB assumption violated exactly where shrinkage matters most) — that extension is open; plus IC/hyperparameters estimated on executable observations only. Chinese public practice (Barra CNE5/Huatai) shrinks risk, never alpha, and handles limits ad hoc in the optimizer.",
"verdict": "Shrinkage per se: DEAD (highest-threat lane; Grinold does it in practitioner clothing, Efron/Harvey-Liu in statistics clothing). Censoring-aware likelihood inside the EB posterior + disagreement-based per-stock variance: SAFE. Threat: HIGH unless framed as censoring-coupled."
}
],
"dedup_verdict": {
"composition_claim_safe": true,
"any_single_work_combines_all_three": false,
"closest_single_work": "Du 2025 (arXiv:2507.07107) — the only found work touching all three slots, but each at the crude level our paper starts from: labels handled by DELETION (bidirectional drop-mask, explicitly not a loss modification), feasibility handled as an execution-time FILTER (optimizer fed tradable names only), shrinkage applied only to the COVARIANCE (Ledoit-Wolf), never to scores/posterior. No observed-indicator correction, no MNAR argument, no executable-return training objective, no extreme-score shrinkage.",
"other_partial_combinations": "JKMP (RFS): implementable objective + end-to-end weights, but smooth convex costs, no hard censoring, no label censoring, no selection shrinkage. BPQP (NeurIPS 2024): end-to-end A-share MVO, static constraint set, limits ignored in both label and action. FR-LUX: frictions-in-reward RL, continuous costs only. No work in DFL, censored regression, survival, bandits, EB/post-selection inference, or the Chinese practitioner corpus couples censored-label supervision + observed-feasible-set decision + posterior shrinkage.",
"required_wording": "Claim the COMPOSITION, not the components: 'first to treat A-share price limits as correlated, observable censoring of BOTH labels and actions, and to optimize shrunk-posterior expected executable return over the induced feasible set.' Components must be presented as adopted (Tobit/Heckman, SPO+/BPQP layer, Tweedie/EB) with Du (deletion), Grinold scale-and-trim (censoring-blind shrinkage), and BPQP (static-constraint end-to-end) as the three baselines beaten.",
"residual_risks": "Leippold-Wang-Zhou (JFE 2022) robustness section on limit-hit exclusion could not be text-extracted — manually verify before claiming no prior label treatment there. Grabit weakens any 'Tobit-in-finance-ML is new' phrasing — avoid it. Novy-Marx-Velikov bands and Yang-Zhang exclude-extremes must be cited to pre-empt the 'mid-band exists' referee."
},
"top3_upgrades": [
{
"rank": 1,
"name": "EM-Tobit ridge label correction (C1: mask -> correction)",
"score_rationale": "Highest gain (recovers exactly the deleted samples at signal extremes, which are the informative ones and Du's own ablation shows dominate Sharpe), lowest cost (wraps the existing ridge solver, no architecture change), highest paper value (the exact line differentiating us from Du: correction vs deletion under MNAR).",
"implementation_sketch": "In engine/executable_strategy.py, keep the Ridge-on-cluster-embeddings model but replace the tradable-label mask with a two-sided Tobit EM loop: (1) fit ridge on currently-imputed labels (init: observed y for tradable, limit value for censored); (2) E-step: for each limit-up sample set y_hat = f(x) + sigma x mills(( +limit - f(x))/sigma) using the truncated-normal conditional mean above the KNOWN per-stock-day threshold (10/5/20/30% by board/ST), mirrored for limit-down; sigma estimated from tradable residuals (optionally per-cluster for heteroscedasticity); (3) M-step: refit ridge on all samples with imputed labels; iterate 3-5 times (converges fast for linear models). Censored samples now contribute shrunk-but-nonzero information instead of vanishing. For any GBDT members of the 9-model zoo, inject the same likelihood as a Grabit-style custom objective (gradient/hessian of the censored log-lik with per-sample bounds). Ablation: mask vs EM-Tobit vs EM-Tobit+Heckman-selection-term, evaluated on executable return."
},
{
"rank": 2,
"name": "Censoring-aware Tweedie posterior scores with disagreement variance (C3+C2 bridge: replaces both equal-weight consensus and band 60-90)",
"score_rationale": "Low cost (pure post-processing of scores, no retraining), high paper value (makes the mid-band endogenous and kills the 'this is just Grinold trimming' objection via the censoring-aware likelihood), solid expected gain (data-adaptive nonlinear tail shrinkage vs a fixed 60-90 band).",
"implementation_sketch": "At each rebalance date: (1) per-stock consensus mean z_i and disagreement variance s_i^2 across the 9 model scores (already computed for consensus — free); (2) fit the cross-sectional log marginal density l(z) of that day's z's by Lindsey's method (Poisson GLM on histogram counts with a 5-7 df natural spline); (3) corrected score z_i + s_i^2 x l'(z_i) — extremes and high-disagreement names shrink hardest automatically; (4) censoring-awareness: for names limit-locked in the label window, inflate s_i^2 by the truncated-normal variance correction (their z is fit on censored info), and estimate the rolling IC used for final scaling on EXECUTABLE observations only; (5) select top-k of corrected scores over the feasible set, replacing the 60-90 band. Ablation grid: fixed band vs Grinold IC-scale+3-sigma-winsorize (the mandatory baseline) vs plain Tweedie vs censoring-aware Tweedie."
},
{
"rank": 3,
"name": "Feasibility shadow price + executable-return objective (C2: band -> feasibility-priced selection)",
"score_rationale": "Moderate cost (needs a new supervised head plus objective change), essential paper value (completes the composition claim — without it the paper is estimation-only), gain concentrated at extremes where P(limit-hit) is largest.",
"implementation_sketch": "(1) Train a supervised feasibility head P(limit-hit at t+1 | x) on the observed censoring indicator (it is data, not a prediction target elsewhere) using cluster embeddings + Liang-2025-style features (recent max return, limit-hit frequency, 一字板 vs 换手板 graded feasibility); (2) selection score becomes corrected_score_i - lambda x P_hat(limit-hit)_i with lambda fit by grid or online (bandits-with-knapsacks shadow-price recipe), and require a pessimistic (lower-confidence) feasibility bound for inclusion; (3) evaluate and select on EXECUTABLE return: zero the box constraint on the locked side for names censored at decision time (observed), charge next-day-open slippage for intended-but-unfilled trades as a post-hoc-regret penalty (NeurIPS 2023 two-stage structure); (4) optional end-stage: swap top-k for a small QP layer (cvxpylayers, BPQP backward if slow) to enable the full end-to-end ablation vs BPQP-with-static-constraints; (5) validation: Lakkaraju-style contraction across limit-width regimes (5% ST / 10% main / 20% ChiNext-STAR, plus 2020 rule change) as the quasi-experiment that the correction is right."
}
],
"must_cite": [
{"paper": "Du, Y. — 'ML Enhanced Multi-Factor Quantitative Trading... with Bias Correction', arXiv:2507.07107 (2025)", "why": "Closest prior art (VERIFIED): same market and censoring problem, deletion-mask labels, publishes the nominal-vs-executable gap (IC +18%, Sharpe -0.44); our correction-vs-deletion delta is defined against it."},
{"paper": "Jensen, Kelly, Malamud, Pedersen — 'Machine Learning and the Implementable Efficient Frontier', RFS (2022/2026)", "why": "Owns the 'train on the implementable objective' thesis at top venue; our objective is theirs specialized to hard observed censoring of labels AND actions."},
{"paper": "Efron, B. — 'Tweedie's Formula and Selection Bias', JASA 106:496 (2011)", "why": "The C3 estimator: posterior-mean correction z + sigma^2 l'(z) that endogenizes extreme-score shrinkage; our extension is the censoring-aware likelihood."},
{"paper": "Sigrist & Hirnschall — 'Grabit: Gradient Tree-Boosted Tobit Models', J. Banking & Finance (2019)", "why": "Closest prior for censored-likelihood training inside modern ML on a finance problem; forces honest framing of C1 as adopted machinery and provides the GBDT injection path."},
{"paper": "Hu, Lee et al. / BPQP — 'BPQP: Differentiable Convex Optimization for End-to-End Portfolio Learning', NeurIPS 2024 (arXiv 2411.19285)", "why": "End-to-end A-share portfolio learning in Qlib at a top venue with static constraints and no limit awareness — the mandatory 'why not just BPQP' baseline and the fast backward pass for our feasibility-parameterized layer."}
],
"runner_up_citations": "Liang (J. Forecasting 2025, P(limit-hit) predictability); Elmachtoub-Grigas SPO+ (Mgmt Sci 2022); Novy-Marx-Velikov (RFS 2016, cost-hysteresis bands to differentiate the mid-band); Harvey-Liu (RFS 2018, EB alpha shrinkage); Pearce et al. (NeurIPS 2022, censored quantile NN); Daskalakis et al. (COLT 2019, known-truncation identifiability); Liu-Wu-Zhu (Econ. Modelling 2022, limit-day signal cleaning); Leippold-Wang-Zhou (JFE 2022, benchmark whose labels ignore censoring — verify robustness section manually)."
}
```
