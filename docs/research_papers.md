# The research papers behind the design, explained simply

Reviewed: 2026-09-26. The original PDFs are kept locally and are not redistributed here; each section links to the public paper record.

This review covers the papers' main methods, experiments, limitations, and relevant appendix material. Key result tables were also inspected visually. This is a reading and design review, not an independent reproduction of the experiments or a formal audit of every proof. Numbers below are **the authors' reported results**, not results from our product. Page references use the PDF page count.

## 1. AdaRubric

**Full title:** AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning. Liang Ding. Supplied version: arXiv:2603.21362v3, 12 pages.

[Paper record](https://arxiv.org/abs/2603.21362)

**The problem:** One checklist does not fit every task. A coding answer should be judged on whether the code works. A research answer should be judged on evidence and coverage. Polished wording cannot make a broken solution correct.

**The idea:** Generate a checklist for the task, score each step against its relevant criteria, and use the resulting evaluations to create training preferences. A *trajectory* means the sequence of an agent's actions and observations. The paper mostly evaluates these sequences, whereas our initial product evaluates final answers plus observable tool results.

**How it works:** The method creates task-specific dimensions and scoring descriptions, judges steps on a 1–5 scale, combines scores, and filters low-quality examples. Its dimension-aware filter requires minimum performance on each specified dimension, so a high average cannot hide an unacceptable weakness. It uses the resulting preferred/dispreferred pairs to improve an agent through preference training.

**What the experiments report:** Table 1 on page 5 gives human-score correlations of 0.79 on WebArena, 0.74 on ToolBench, and 0.77 on AgentBench; the reported average is 0.77. Correlation measures how scores move together; 0.79 does not mean 79% of answers are correct. Table 2 reports task-success increases of 6.8, 8.5, and 7.7 percentage points over its Prometheus comparison after DPO training. Table 3 gives inter-run agreement of 0.85 and 0.82 on two benchmarks, averaging 0.84.

**Important limits:** Its human correlation study uses 300 trajectory pairs per benchmark and three annotators. Pages 9 and 11 discuss incomplete rubrics, model-predicted confidence, and substantial cost: roughly 40 evaluator calls for an eight-step, five-dimension example. Repeated evaluators agreeing does not establish truth. The abstract's broad 0.79/0.83 summary differs from the table aggregates; our notes retain the table-specific values. The headline improvements cannot be transferred to our product without testing.

**What we adopt:** Start with short, manually understandable checklists for factual answers, code, and math. Keep central correctness checks separate from style. Cache reusable checklists. Later compare fixed task templates against generated task-specific checklists under the same spending allowance.

**What we defer:** Forty-call grading, per-step reward training, and DPO. DPO teaches a generator to prefer certain outputs; it is not the required training recipe for our answer judge. Do not implement the paper's formulas from an abstract or assume its proposed agreement threshold is a launch rule for us.

**Small experiment:** Compare a general “pick the best answer” judge with a source-aware judge using a task checklist. Measure both against independent human labels.

## 2. Optimal Model Activation Policies

**Full title:** Optimal Model Activation Policies for Inference Networks of Large Language Models. Foivos Charalampakos, Md Ibrahim Ibne Alam, Iordanis Koutsopoulos, and Koushik Kar. Supplied document: 46 pages including supplementary material; arXiv:2609.15992v1.

[Paper record](https://arxiv.org/abs/2609.15992)

**The problem:** Calling the expensive model for every question costs money or compute even when a cheaper model can answer adequately.

**The idea:** Ask a cheaper model first. If a measured confidence signal is below a threshold, ask a more expensive model. Learn the threshold from examples with known outcomes. For classification, different categories can have different thresholds.

**Simple example:** If the cheaper model answers a straightforward question correctly and the checks pass, stop. If its answer lacks support or fails a check, spend the remaining budget on another attempt. A *threshold* is the boundary separating these decisions.

**What it actually studies:** The theoretical development focuses on a two-model chain. It considers both an ideal second model and a second model that can also make mistakes. Optimality within the chosen threshold family is distinct from optimality among every possible routing strategy. The broader threshold result depends on assumptions such as higher confidence corresponding to lower error risk; the non-oracle treatment also simplifies the second model's conditional errors. See section 3 and appendices C–D.

**What the experiments report:** Page 7, Table 1 reports 26.66% compute-cost savings on SQuAD for a Gemma3-1B/Gemma3-4B pair, with ROUGE-L 0.57 and cosine similarity 0.64 versus 0.57/0.65 for full deferral. Its striking 97% WMT saving occurs where the larger model performs worse. These are particular experiments using profiled compute costs, not promised API bill reductions. Data is split 80/20 for optimization/testing, with samples limited by compute.

**Important limits:** Text overlap and embedding similarity can miss factual contradictions. Token confidence measures how expected the words are, not whether they are true. API providers may expose different probability information, and prices depend on token usage and tools. A threshold chosen on historical data may fail when models or questions change. Five providers running in parallel are not the two-model chain proved in this paper.

**What we adopt:** Budget limits, measured escalation rules, and explicit quality-versus-cost comparisons. Evaluate both two-model parallel comparison and a cheaper-first cascade; neither wins by definition.

**What we defer:** Claims of mathematically optimal general routing, complex graph searches, and a trained router before we have outcome logs.

**Small experiment:** Compare one strong model, always calling two models, and cheaper-first escalation on the same questions. Include all checking and retrieval costs.

## 3. PDE-Agents, stored as “LLM Orchestrated”

**Full title:** PDE-Agents: An LLM-Orchestrated Multi-Agent Framework for Automated Finite Element Simulations with Knowledge Graph-Augmented Reasoning. Sayan Adhikari, Gulshan Noorsumar, and Øyvind Jensen. Supplied version: arXiv:2606.07850v2, 20 PDF pages.

[Paper record](https://arxiv.org/abs/2606.07850)

**The problem:** An AI can successfully run an engineering simulation using invented material properties. The software finishes, but the physical result is wrong.

**The idea:** Separate planning, simulation, analysis, and stored knowledge. Give the model relevant known information early. Use a fixed numerical solver and check its outputs. A *knowledge graph* is a store of connected facts; *GraphRAG* retrieves information from that store to help the model.

**The three conditions:** “KG Off” omits graph access. “KG On” requires graph-related tool calls. “KG Smart” supplies relevant previous examples before the agent loop and allows additional retrieval when useful. All three use the same frozen knowledge snapshot in the main comparison. Freezing prevents one run from teaching a later condition during that comparison.

**What the experiments report:** Page 10, Table 4 compares 50 tasks per condition. Both KG Off and KG Smart complete 100%; KG On completes 94%. But the material-property fidelity score is 0.796 for KG Off and 0.926 for KG Smart. For fictional materials whose values are provided through the knowledge base, the figures are 0.340 and 1.000. The system can finish while using incorrect information. The reported success-rate gap is not statistically significant at the usual 0.05 level: the paper gives p = 0.079.

**Important limits:** This is a narrow heat-simulation system using a preverified solver, not a general answer-ranking product. It does not prove that a knowledge graph beats simple passage retrieval for our use case. The authors attribute much of the improvement to early context provision; they do not independently vary every component of KG Smart. The reported 97.8% across 1,369 operational runs mixes activity types and is not a clean general-Q&A accuracy measure. Citation traceability is future work, and page 16 says its agents currently run sequentially.

**What we adopt:** Separate tools from judgment; check inputs as well as outcomes; provide relevant evidence before generation when needed; keep source/run records; compare retrieval approaches using a frozen corpus.

**What we defer:** Neo4j, ten-service deployment, engineering solvers, and a large agent framework. A small passage store is enough to test our first evidence pipeline.

**Small experiment:** Compare no evidence, a fixed relevant evidence package, and an automatically retrieved package. This reveals whether the bottleneck is finding evidence or using it correctly.

## 4. Judge Anything

**Full title:** Judge Anything: MLLM as a Judge Across Any Modality. Shu Pu, Yaochen Wang, Dongping Chen, and collaborators. KDD 2025, 12 pages. DOI: 10.1145/3711896.3737409.

[Authors' project](https://urrealhero.github.io/judgeanythingweb/)

**The problem:** A model that understands or generates content is not necessarily good at judging it, especially across text, image, audio, and video.

**The idea:** Test the judges directly against human judgments. TaskAnything contains 1,500 queries across 15 task types. JudgeAnything adds 6,000 individual-response scoring cases and 3,000 pair-comparison cases. It compares general judgment, rubric-guided judgment, and more specific checklists.

**Simple distinction:** Asking “which of these two answers is better?” is a different task from assigning a score from 1 to 5. Neither one automatically tells us whether both answers are bad.

**What the experiments report:** Table 2 on page 5 reports 70.60% pair-comparison agreement including ties for Gemini-1.5-Pro using checklists on multimodal understanding tasks. That is a historical result in one setting, not a current model recommendation. The study finds weaker performance in many generation settings and shows that extra rubric detail can help some judges while harming others. The abstract's 66.55% and 53.37% figures correspond to the evaluator-fusion checklist row in Table 2; they should not be treated as a universal average accuracy for all judges.

**Important limits:** Agreement with human preferences is not a proof of factual truth. Human labels and checklists can be subjective. Strong performance in text does not establish competence in video or audio. This benchmark is useful for evaluation research, but is not a ready-made evidence-conditioned text-judge training curriculum.

**What we adopt:** Evaluate our judge separately; allow ties and insufficient-evidence outcomes; inspect failures; test whether checklist complexity helps. Hide provider labels and swap answer positions as additional controls in our design, without claiming that these eliminate every bias.

**What we defer:** Image/audio/video generation and universal multimodal scoring.

**Small experiment:** Give a judge the same pair in A/B and B/A order. Measure whether it changes its choice merely because the positions changed. Include two good answers and two bad answers.

## How these papers change our project

| Lesson | Concrete project decision | Test before trusting it |
| --- | --- | --- |
| AdaRubric: match checks to tasks | Task-specific rubrics and minimum correctness requirements | Compare with a simple general judge at matched cost |
| Activation policies: spend selectively | Bounded escalation, then a learned router | Measure quality, dollars, and wait time together |
| PDE-Agents: completion differs from correctness | Evidence before grounded generation and separate verifiers | Use both known evidence and retrieved evidence conditions |
| Judge Anything: judges have weaknesses | Independent judge evaluation and failure analysis | Human review, answer-order swaps, ties, and incorrect candidates |

## Our research hypothesis

For a clearly defined group of questions, a judge given relevant evidence and verifier results may select better-supported answers than a judge given only candidate answers, at an acceptable additional cost.

That is our hypothesis, not a proven novelty claim. We will first establish a baseline, then change one component at a time. To claim SOTA later, we would need named benchmarks, comparable current baselines, untouched tests, reproducible settings, uncertainty estimates, and a broader literature review than these four papers. A useful product can succeed without a universal SOTA claim.
