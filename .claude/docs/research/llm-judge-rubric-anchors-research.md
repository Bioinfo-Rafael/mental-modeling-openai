# Anchored 1-5 Rubrics for LLM-as-a-Judge and Human Rating

Research date: 2026-11-04. Scope: how to write per-level anchors and how to make
an LLM judge use them, for scoring LLM-written explanations of an air-combat RL
agent's intent on four aspects (completeness of required elements, correctness
against records, timing, contribution to human decisions), 1-5 plus
"cannot judge" (CJ) and "not applicable" (NA), rated by one human and one LLM judge.

Legend: **[V]** verified against the primary source (abstract, paper HTML, or
official doc) in this session. **[S]** taken from a secondary summary, not
checked against the primary text. **[U]** unverified or from memory.

---

## 1. Writing per-level anchors (human-rating literature)

### 1.1 Behaviorally anchored rating scales (BARS)

- Origin: Smith & Kendall (1963), "Retranslation of expectations: An approach to
  the construction of unambiguous anchors for rating scales", *J. Applied
  Psychology* 47:149-155, doi:10.1037/h0047060. [S]
  https://en.wikipedia.org/wiki/Behaviorally_anchored_rating_scales
- Procedure [S]: (1) domain experts write critical incidents, i.e. concrete
  observed examples ranging from very poor to very good; (2) incidents are
  grouped into dimensions; (3) **retranslation**: a second, independent group of
  experts re-assigns each incident to a dimension and a scale value; incidents
  with low agreement on dimension or high variance on scale value are dropped;
  (4) surviving incidents become the anchors at their mean scale value.
  Exact step list: USAF report https://apps.dtic.mil/sti/pdfs/ADA239969.pdf
  (not retrieved).
- Evidence on benefit: reviews found BARS "no better or worse than other methods"
  on quantitative psychometric criteria, but better on use and acceptance
  criteria (Jacobs, Kafry & Zedeck review, as quoted). Rater training reduced
  rating errors more reliably than changing the scale format. [S]
  https://pure.psu.edu/en/publications/expectations-of-behaviorally-anchored-rating-scales/ ,
  https://apps.dtic.mil/sti/pdfs/ADA142009.pdf
- Takeaway for us: the value of BARS is the *process*: anchors are built from
  real examples and checked by an independent sort. We can do this cheaply.
  Draft anchors, then have a second person (or the judge) blind-assign a set of
  real explanations to levels and rewrite any anchor that people disagree on.

### 1.2 Analytic vs holistic rubrics

- Brookhart (2018), "Appropriate criteria: key to effective rubrics", *Frontiers
  in Education* 3:22. A rubric = criteria + performance-level descriptions; both
  are needed. In 7 of 51 higher-education rubrics reviewed, the level
  descriptions used rating-scale language ("good", "fair") or counted
  occurrences instead of describing quality. Criteria should describe the
  quality of the target, not compliance with instructions. [S]
  https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2018.00022/epub
- Jonsson & Svingby (2007), *Educational Research Review*, review of 75 studies:
  reliability is associated with analytic criteria, task-specific wording,
  exemplars, and rater training. [S] (reported via
  https://www.koji.so/visuals/scoring-rubric-reliability-conditions and
  https://www.frontiersin.org/articles/10.3389/feduc.2019.00089)
- Holistic vs analytic evidence on agreement is mixed. Holistic marks sometimes
  show higher cross-marker agreement. Rater experience can matter more than
  rubric type (Chamberlain et al. 2026, Frontiers). [S]
  https://www.frontiersin.org/articles/10.3389/feduc.2019.00089 ,
  https://eprints.whiterose.ac.uk/id/eprint/240546/1/feduc-11-1729644.pdf
- Takeaway: our four aspects should be scored analytically, one score per
  aspect. Each aspect needs task-specific wording, worked examples, and a short
  calibration session for the human rater.

### 1.3 Observable, monotone, mutually exclusive levels (synthesis)

There is no single primary source for this triple. It is the common thread of
BARS (observable behaviour), Brookhart (describe quality, not "good/fair"), and
rubric guides that require "observable and measurable" differences between
levels (https://www.bristolcc.edu/_resources/documents/Rubric_Design_Guide.pdf, [S]).
In practice:

- **Observable**: each anchor names something checkable in the explanation or in
  the episode record ("states the target track", "maneuver named matches the
  logged action"), not an impression ("clear", "good").
- **Monotone**: the same underlying variable increases from 1 to 5. Do not
  change what is measured between levels (e.g. level 3 about count of elements,
  level 4 about wording).
- **Mutually exclusive and exhaustive**: every possible explanation fits exactly
  one level. Tie-break rules decide borderline cases ("if two conditions hold,
  take the lower level"). CJ and NA sit outside the scale and have their own
  definitions.
- Counting is acceptable when the aspect *is* about coverage. Brookhart's
  criticism targets counting used as a proxy for quality. "Completeness of
  required elements" is legitimately a coverage count, but anchors should say
  *which* elements count and what makes an element "present".

---

## 2. LLM-as-a-judge findings

### 2.1 Rubric structure: per-criterion and per-instance anchors

- **Prometheus** (Kim et al., arXiv 2310.08491, ICLR 2024) [V]: the rubric is a
  criterion description plus a description for each score 1-5. The model writes
  feedback first, then `[RESULT] n`. Ablation (Table 6, Pearson, 7B):
  - full model: 0.860 (seen rubrics), 0.847 (unseen), 0.457 (Vicuna Bench);
  - without the score rubric: 0.837, 0.745, 0.355;
  - without the reference answer: 0.642, 0.626, 0.349.

  The reference answer contributed most. The rubric mattered most on unseen
  criteria.
  https://arxiv.org/html/2310.08491v2
- **Prometheus 2** (Kim et al., arXiv 2405.01535, May 2024) [V, abstract and
  model card]: same input format (instruction, response, reference answer
  labelled as score 5, criterion, `score1_description` ... `score5_description`).
  It supports direct assessment and pairwise ranking.
  https://arxiv.org/html/2405.01535v2 , https://huggingface.co/kaist-ai/prometheus-8x7b-v2.0
- **FLASK** (Ye et al., arXiv 2307.10928, ICLR 2024) [S]: 12 fine-grained skills,
  each scored 1-5 with a skill-specific rubric. Instance-specific rubrics are
  added for the hard subset. Fine-grained scoring raised human correlation
  compared with one overall score. https://arxiv.org/html/2307.10928v3
- **"Rubric Is All You Need"** (Pathak et al., arXiv 2503.23989, ICER 2025) [V,
  abstract]: question-specific rubrics substantially beat question-agnostic
  rubrics for LLM grading. The paper also introduces a "Leniency" metric for
  judge strictness relative to experts. https://arxiv.org/abs/2503.23989v3
- **HealthBench** (Arora et al., OpenAI, arXiv 2505.08775, May 2025) [V,
  abstract]: 48,562 conversation-specific criteria written by 262 physicians.
  Per-criterion grading with binary met/not-met and a reported grader macro F1
  of about 0.71 versus physicians are [S] only.
  https://arxiv.org/abs/2505.08775
- **G-Eval** (Liu et al., arXiv 2303.16634, EMNLP 2023) [V via ar5iv]: the prompt
  has a task introduction, criteria, and LLM-generated evaluation steps, then a
  form ("Coherence:"). Coherence is scored 1-5, dialogue engagingness 1-3. The
  final score is sum p(s_i) * s_i over score tokens. This avoids ties and one
  dominant integer; for GPT-4 the probabilities were estimated from 20 samples.
  G-Eval rated GPT-3.5 summaries above human-written ones that humans
  preferred. https://ar5iv.labs.arxiv.org/html/2303.16634

### 2.2 Checklists and decomposition

- **CheckEval** (Lee et al., arXiv 2403.18771, v3 Oct 2025) [V, abstract]: humans
  choose dimensions and sub-dimensions; these become Boolean yes/no questions.
  Across 12 judge models this gave higher agreement between judges and lower
  variance than direct scoring. https://arxiv.org/abs/2403.18771
- **TICK** (Cook et al., arXiv 2410.03608, Oct 2024) [V, abstract]: the LLM writes
  instruction-specific yes/no checklists. Exact agreement with human preference
  rose from 46.4% to 52.2%. Giving the same checklists to *human* raters raised
  their inter-annotator agreement from 0.194 to 0.256.
  https://arxiv.org/abs/2410.03608
- **LLM-Rubric** (Hashemi et al., ACL 2024) [V, abstract]: 9 rubric questions.
  The LLM's probability distribution over answers is fed into a small
  calibration network with judge-specific parameters, which predicts human
  ratings (1-4 scale, RMSE < 0.5, 2x better than uncalibrated).
  https://aclanthology.org/2024.acl-long.745
- **Anthropic, "Demystifying evals for AI agents"** (2026-01-09) [V]: "grade each
  dimension with an isolated LLM-as-judge rather than using one to grade all
  dimensions"; "vague rubrics produce inconsistent judgments"; "give the LLM a
  way out, like providing an instruction to return 'Unknown' when it doesn't
  have enough information"; "For tasks with multiple components, build in
  partial credit"; judges "should be closely calibrated with human experts".
  https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- **Anthropic docs, "Develop tests"** [V, page undated]: "Have detailed, clear
  rubrics"; "Be empirical or specific", e.g. correct/incorrect or 1-5; "Encourage
  reasoning", i.e. use a grader with thinking on so it reasons before scoring;
  "Generally best practice to evaluate with a different model from the one used
  to generate the evaluated output".
  https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
- **OpenAI eval guidance**: not retrieved from platform.openai.com. Secondary
  summaries mention reference-guided and pairwise graders, clear rubrics, and
  length control [U]. Treat OpenAI-specific claims as unverified.

### 2.3 Rationale before score; "why n and not n+1"

- Chiang & Lee (arXiv 2310.05657, Findings EMNLP 2023) [S]: asking the judge to
  explain or analyse its rating consistently improved correlation with humans
  and robustness to temperature. G-Eval's auto-CoT steps did not reliably help.
  The study also included rate-then-explain variants.
  https://arxiv.org/abs/2310.05657v1
- Chen, Chu & Nakayama (arXiv 2406.02863, 2024) [V, abstract]: the order of
  reason and score "significantly influences" scores; reason-first gave more
  comprehensive evaluations. One summary says it also gave higher scores [S].
  https://arxiv.org/abs/2406.02863v1
- Prometheus is trained feedback-first, and Anthropic recommends reasoning
  before the score (above).
- **"Why level n and not n+1"**: no primary study found that tests this prompt.
  It is a reasonable adaptation of the boundary and tie-break idea from BARS and
  rubric design. Contrastive rubric generation (CDRRM, arXiv 2603.08035) is
  related but works on pairs, not adjacent levels [S]. Treat it as an
  untested design choice and check it against human agreement.

### 2.4 Scale granularity

- Li et al., "Grading Scale Impact on LLM-as-a-Judge" (arXiv 2601.03444,
  2026-01-06) [V, abstract]: across three scales and six benchmarks, a 0-5 scale
  gave the highest human-LLM absolute agreement (ICC). The scale choice shifted
  agreement "substantially" on subjective tasks.
  https://arxiv.org/abs/2601.03444
- Rubric-Conditioned LLM Grading (arXiv 2601.08843) [S]: alignment is strong on
  binary tasks and degrades as rubric granularity increases.
  https://arxiv.org/html/2601.08843v1
- Salamandra report (arXiv 2502.08489) [S]: for some criteria, 5 levels were too
  fine "even [for] expert humans". https://arxiv.org/pdf/2502.08489
- Takeaway: 1-5 is defensible if each level is anchored. Collapse to 1-3 for an
  aspect where pilot humans cannot separate adjacent levels. Binary checklist
  sub-items under each aspect reduce variance (CheckEval, TICK).

### 2.5 Known biases and mitigations

- **Position bias** (Zheng et al., MT-Bench, arXiv 2306.05685, NeurIPS 2023) [V]:
  verdicts were consistent after swapping answer order in only 65.0% of cases
  for GPT-4, 46.2% for GPT-3.5, and 23.8% for Claude-v1. The fix is to judge both
  orders and count a win only if the verdicts agree.
  https://arxiv.org/html/2306.05685v4
- **Rubric option position bias**: Xu, Hirasawa, Kozuno & Ushiku, "Am I More
  Pointwise or Pairwise?" (arXiv 2602.02219, Findings EMNLP 2026) [S]. Judges
  favour score options at certain positions, mostly the first. The order of
  criteria in a multi-criterion prompt also shifts scores. Balanced permutation
  of option order reduces the bias and improves human correlation; a few
  permutations are enough. https://arxiv.org/abs/2602.02219
- **Scoring-prompt bias**: Li et al., "Evaluating Scoring Bias in
  LLM-as-a-Judge" (arXiv 2506.22316, DASFAA 2026) [V, abstract]. Three biases:
  rubric order, score ID, and reference-answer score. A secondary summary says
  the paper recommends descending rubric order, letter or Roman IDs, and a
  full-mark reference answer [S]. https://arxiv.org/abs/2506.22316
- **Verbosity bias** (Zheng et al.) [V]: under a "repetitive list" attack, GPT-3.5
  and Claude-v1 failed 91.3% of the time and GPT-4 failed 8.7%.
- **Self-preference**: Panickssery, Bowman & Feng (arXiv 2404.13076, NeurIPS
  2024) [V, abstract]. Judges rate their own outputs higher than equal-quality
  outputs, and this correlates linearly with how well they recognise their own
  text. G-Eval also favoured LLM-written text. https://arxiv.org/abs/2404.13076v1
- **Familiarity and anchoring**: Stureborg et al. (arXiv 2405.01724, 2024) [S].
  Judges prefer low-perplexity text and use some score values much more often
  than others. When several labels are produced in one output, earlier labels
  anchor later ones. https://arxiv.org/abs/2405.01724
- **Misled by the answer under review** (Zheng et al.) [V]: GPT-4 accepted wrong
  math answers 14 of 20 times. Giving a reference answer cut failures to 3 of 20.
- **Leniency**: Thakur et al., "Judging the Judges" (arXiv 2406.12624) [V,
  abstract and summary]. Judges tend toward leniency, and high percent agreement
  can hide large score differences. Cohen's kappa separates judges better, but
  "is not all telling". https://arxiv.org/html/2406.12624v2

### 2.6 Calibration against human ratings

- Metric: for ordinal 1-5, use **quadratic-weighted Cohen's kappa** together with
  exact and within-1 agreement, a confusion matrix, and the mean signed
  difference (leniency). Thakur et al. show why percent agreement alone
  misleads. Li et al. use ICC for absolute agreement.
- Exclude CJ and NA from kappa, and report them separately: the rate per rater,
  and agreement on *whether* an item is CJ or NA (unweighted kappa on a
  three-way label: scored / CJ / NA). This exclusion is our design choice, not
  taken from a source.
- **Rating indeterminacy**: Guerdan et al. (arXiv 2503.05965, NeurIPS 2025) [V,
  abstract]. When more than one rating is defensible, forced-choice validation
  can pick the wrong judge; under multi-label validation, the best forced-choice
  judges can be among the worst. Mitigation: let the human mark a *set* of
  acceptable levels (e.g. "3 or 4") on ambiguous items.
  https://arxiv.org/abs/2503.05965
- **Sample size** [S]: Sim & Wright (2005, *Physical Therapy* 85:257) tabulate
  sample sizes for kappa studies (tables not retrieved).
  https://pubmed.ncbi.nlm.nih.gov/15733050/
  Wald confidence intervals for kappa behave properly only from about n = 100
  items (Lee & Tu 1994, as cited). Bootstrap intervals should not be used below
  about 35 items (Klar et al. 2002, as cited).
  https://escholarship.org/content/qt33b7w4bw/qt33b7w4bw.pdf
  An unsourced vendor heuristic says 100-200 items for a tight CI at moderate
  agreement [U]. Practical floor: about 50 doubly-rated items per aspect for a
  pilot and 100 or more for a reportable kappa with a bootstrap CI.
- **Bar to compare against**: judge-human agreement should be judged against
  human-human agreement on the same items. HealthBench and JUDGE-BENCH
  (Bavaresco et al., ACL 2025, https://aclanthology.org/2025.acl-short.20) [V,
  abstract] both frame validity this way. JUDGE-BENCH found reliability varies
  by task and property, and must be validated per task. With one human rater
  there is no human-human baseline, so plan a second human on a subset (≥ 30
  items) if possible.

---

## 3. Explanation-quality scales from XAI

- **Hoffman, Mueller, Klein & Litman**, "Metrics for Explainable AI: Challenges
  and Prospects" (arXiv 1812.04608, 2018), published as "Measures for
  explainable AI ..." in *Frontiers in Computer Science* 5, 2023-02-06 [V].
  https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2023.1096257/full
  - **Explanation Goodness Checklist** (a priori, yes/no, completed by
    independent judges, not the developers): understand how it works,
    satisfying, sufficiently detailed, sufficiently complete, actionable (helps
    me know how to use it), tells how accurate or reliable, tells how
    trustworthy.
  - **Explanation Satisfaction Scale** (a posteriori, users, 5-point agree to
    disagree): know how it works, satisfying, sufficient detail, complete,
    tells me how to use it, useful to my goals, shows how accurate.
  - Caveat from the authors: users can report satisfaction while their
    understanding is partial or wrong. Mental-model and task-performance
    measures (prediction accuracy, task success) are needed alongside it.
- **Belouadah et al.**, "Evaluating the effectiveness of LLMs for explainable deep
  reinforcement learning", *Machine Learning with Applications* 22:100795,
  2025-11-14 [V, abstract]: an LLM judge of textual explanations of a DRL / MCTS
  agent agreed with humans at weighted kappa = 0.77 and Spearman rho = 0.88.
  Rubric dimensions not retrieved. https://orbilu.uni.lu/handle/10993/66628
- Related XRL text-explanation work, with rubrics not retrieved [S]: Terra et al.
  (arXiv 2601.02514, textual policy explanations); SySLLM (arXiv 2503.10509);
  TalkToAgent (arXiv 2509.04809); RL failure explanations scored on coherence,
  helpfulness, and stability, using environment obfuscation to test grounding
  (https://ceur-ws.org/Vol-4254/paper15.pdf).
- Mapping to our aspects:
  - Goodness "complete" and "detailed" map to completeness.
  - "Accurate or reliable" maps to correctness against records. Our version is
    stronger because it is checked against logs, not perceived.
  - "Actionable" and "useful to my goals" map to contribution to human
    decisions.
  - Timing has no XAI-scale counterpart and needs its own anchors.

---

## Recommendations for our rubric

1. Score each aspect in a **separate judge call**: an isolated judge per
   dimension (Anthropic 2026), which avoids anchoring across labels (Stureborg)
   and criteria-order bias (Xu et al. 2026).
2. Write **aspect-specific anchors** for all five levels, not one generic
   1-5 description (Prometheus, FLASK, Rubric Is All You Need). Each anchor
   names an observable condition in the explanation or the episode record. Ban
   "good", "adequate", and "clear" as the only content of an anchor.
3. Keep one variable per aspect so levels are **monotone**. Add an explicit
   tie-break: "if conditions for two levels hold, assign the lower". Make every
   explanation fit exactly one level.
4. Under each aspect, list **binary sub-checks** (e.g. for completeness: target
   identified? maneuver named? goal stated? time horizon given?), and define the
   1-5 level as a function of the sub-checks (CheckEval, TICK, HealthBench). The
   judge answers the sub-checks first, then maps them to a level.
5. For correctness and timing, give the judge the **episode record or extracted
   facts as the reference**. A reference was the largest single gain in the
   Prometheus ablation and cut reference-free failures from 14/20 to 3/20 in
   MT-Bench. Never ask the judge to infer ground truth.
6. Define **CJ** as "the record does not contain the fact needed to verify the
   claim", and **NA** as "the aspect does not arise for this episode or step
   (e.g. no decision point)". Give one example of each, and tell the judge to use
   them instead of guessing (Anthropic "Unknown" escape).
7. Require **evidence, then sub-checks, then score**: quote the span and the record
   field. Use a reasoning or thinking judge (Anthropic docs, Prometheus,
   Chiang & Lee). Store the rationale for audit; parse only the final tagged
   score.
8. Ask for a one-line **"why not n+1 (and why not n-1)"** that cites the failed
   condition of the adjacent anchor. Treat it as untested: keep it only if
   pilot weighted kappa does not drop.
9. Build anchors **BARS-style**: collect about 10-15 real explanations per aspect
   spanning the range, have the human and a second person blind-assign levels,
   and rewrite anchors where they disagree by 2 or more. Put one short worked
   example per level into the judge prompt.
10. Keep **1-5** (0-5 had the best human-LLM agreement in Li et al. 2026). If the
    pilot shows humans cannot separate adjacent levels on an aspect (e.g.
    contribution to decisions), collapse that aspect to 1-3 instead of forcing
    five levels.
11. Control **position and order bias**: list levels in a fixed order (descending
    5 to 1 is suggested by Li et al. 2025 [S]), and run a small balanced
    permutation check of level order on a subset (Xu et al. 2026).
12. Control **self-preference and verbosity**: the judge must be a different model
    (family, if possible) from the explanation generator (Panickssery;
    Anthropic docs). Add an anchor rule that extra text without new required
    content does not raise a level. Track score against length.
13. Calibrate with **quadratic-weighted kappa** per aspect, plus exact and
    within-1 agreement, a confusion matrix, and mean signed difference
    (leniency). Report CJ/NA rates and agreement on CJ/NA vs scored separately,
    excluding CJ/NA from kappa.
14. Size the calibration set at about **50 items per aspect for the pilot and 100+
    for reporting**, with bootstrap 95% CIs. Add a second human on 30 or more
    items so judge-human kappa can be compared with human-human kappa.
15. On items where the human hesitates, let them record an **acceptable set** of
    levels (e.g. {3,4}) and also compute set-based agreement (Guerdan et al.
    2025). Borrow Hoffman's Goodness items ("complete", "actionable", "accurate")
    as wording seeds, but keep the scored anchors record-grounded, because
    satisfaction is not understanding.
