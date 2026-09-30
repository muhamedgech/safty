# PACT — A Book-Style Tutorial

**Parallel-Anchored Conformal Transfer: a cross-lingual safety certificate for LLMs**

> How to read this book. Each chapter has three layers: first the **idea** in plain
> words with a picture or analogy, then the **math** with a tiny worked example you can
> check by hand, then the **code** — the exact function in this repo that does it. If a
> chapter's math feels heavy, read only the idea and the code; you will still understand
> the whole system. Symbols and terms are collected in the Glossary (Appendix A). You do
> not need to read linearly, but Parts I–III build on each other.

---

## Table of contents

- **Part I — The Problem**
  - Ch 1. Why safety breaks when you change language
  - Ch 2. The refuse/comply decision, and how we read it from inside the model
  - Ch 3. The gap PACT fills: transport without a guarantee
- **Part II — The Tools**
  - Ch 4. Activations and linear probes
  - Ch 5. Conformal prediction from scratch
  - Ch 6. From one threshold to a certificate (2-state, 3-state, PAC)
- **Part III — The PACT Method**
  - Ch 7. The three scores: A0, A2, TT
  - Ch 8. Parallel anchoring — why paired translations matter
  - Ch 9. Assembling the certificate
- **Part IV — Evaluation**
  - Ch 10. The datasets
  - Ch 11. Leakage-free protocol (fit / cal / test)
  - Ch 12. The baseline: Aziz few-shot gate
  - Ch 13. The metrics: miss, flag, deferral, AUC floor
- **Part V — Results and Meaning**
  - Ch 14. The six-language result, read slowly
  - Ch 15. What it proves, and what it does not
  - Ch 16. Where PACT sits in the literature (novelty)
- **Part VI — The Code**
  - Ch 17. The `safty` pipeline in 8 steps
  - Ch 18. The `pact_extension` on real activations
  - Ch 19. Running everything
- **Appendices**
  - A. Glossary   B. Symbols   C. FAQ

---

# Part I — The Problem

## Chapter 1. Why safety breaks when you change language

A large language model is trained, mostly, on English. Its safety training —
"refuse to explain how to build a weapon", "answer harmless questions normally" —
is also mostly in English. So the model has a strong, reliable *refuse-or-comply*
reflex **in English**.

Now ask the same dangerous question in Amharic, Khmer, or Yoruba. Two things can go wrong:

1. **Under-refusal (the dangerous failure).** The safety reflex is weaker in the
   low-resource language, so a request the model would refuse in English slips through
   and gets a helpful, harmful answer.
2. **Over-refusal (the annoying failure).** The model becomes jumpy in the other
   language and refuses perfectly safe questions ("how do I *kill* a Python process?").

The core reason is simple: **the model understands the languages unequally, but we ask
it to be equally safe in all of them.** English safety is strong; low-resource safety is
a patchwork. And we usually have *almost no labeled safety data* in those languages to
check or fix the problem.

**The question of this research:** can we take the safety decision the model already
makes well in English and *carry it over* to another language — and, crucially, attach a
**guarantee** to how often it will be wrong there, **without collecting a pile of labels
in that language?**

That guarantee is the whole point. Anyone can build a classifier that is "usually right."
PACT builds one that comes with a *number you can trust*: "in this language, a genuinely
harmful prompt will be let through at most 5% of the time." That number is called a
**certificate**.

---

## Chapter 2. The refuse/comply decision, and how we read it from inside the model

### The idea

When an instruction-tuned model decides whether to refuse, that decision is not only in
its final words — it is already present in its **internal activations** partway through the
network, *before* it has written a single token of the answer. Think of it as the model's
"state of mind" right after reading the instruction: a long list of numbers (a vector)
that already leans toward *refuse* or *comply*.

If we can find the direction in that vector space that means "this is refusal-worthy", we
can read the decision off directly — cheaply, and at a fixed point in the computation.

### A picture

Imagine every prompt becomes a dot in a high-dimensional space (say 4096 numbers for
Llama-3.1-8B). Harmful prompts cluster on one side, harmless prompts on the other. A
**probe** is just a ruler laid across the gap: project a dot onto the ruler, read a number.
High number → looks harmful. Low number → looks safe.

### The code hook

We extract activations at one **layer** (layer 10 for Llama in these experiments) and one
**token position** (`t_post_inst` — right after the instruction is read). That single
vector per prompt is the input to everything else. In the repo, the extraction lives in
`scripts/step05_extract_features.py`; the cached vectors are what every score in Part III
consumes.

> **Key term — activation.** The intermediate output of a layer, a vector of numbers, for
> one input at one token position. It is the model's internal representation of the prompt
> at that moment.

---

## Chapter 3. The gap PACT fills: transport without a guarantee

People already know how to *transport* a probe: train the "refusal ruler" in English,
then apply it to another language's activations. This often works okay. The problem is it
comes with **no promise**. You do not know, for Amharic specifically, whether it misses 3%
of harmful prompts or 30%. And the usual fix — "just label a few hundred Amharic prompts
and retrain" — is exactly the expensive thing we are trying to avoid, and *still* gives no
guarantee.

PACT closes this gap with two moves, which are the two halves of its name:

- **Parallel-Anchored** — it uses *parallel* (translated, aligned) prompts to carry the
  English decision into the target language, and to *align* the target's internal space
  back onto English so the English ruler still reads correctly (Chapters 7–8).
- **Conformal Transfer** — it wraps the transported score in a **split-conformal
  threshold**, which turns "usually right" into "wrong at most α of the time, provably"
  (Chapters 5–6).

The result is a per-language safety gate with a dial you set (α = 0.05, say) and a
guarantee that the miss rate stays at or below that dial — using **zero new labels** in
the target language.

---

# Part II — The Tools

## Chapter 4. Activations and linear probes

### The idea

A **linear probe** is the simplest possible classifier: a weight vector **w** and a bias
*b*. Given an activation **x**, it computes a score. We turn that into a probability with
the logistic (sigmoid) function:

```
p = sigmoid(w · x + b)      # probability the prompt is "safe/comply"
```

We then define the **score** we actually use as

```
s = -p        # high s = looks harmful (refusal-worthy)
```

Why the minus sign? Because the whole certificate is written in terms of "high score =
dangerous." Flipping the sign once here keeps every later formula pointing the same way.
(This exact orientation was a real bug once: if you train the probe to predict *harmful*
instead of *safe*, the score points the wrong way and AUC comes out below 0.5. In the code
we train on the **safe** class, `1 - y`, so high `s` reliably means harmful.)

### Why linear?

Because the safety decision really does live along roughly a *direction* in activation
space — a straight ruler is enough to read it, and a straight ruler is easy to *transport*
and *align* between languages (a curvy classifier is not). Simplicity here is a feature.

### A tiny worked example

Two harmful activations project to scores `0.9, 0.8`; two harmless to `0.1, 0.2`. A ruler
that separates them perfectly gives AUC = 1.0. If instead harmful gave `0.6, 0.4` and
harmless `0.5, 0.3`, they overlap and AUC drops — the probe is weaker, and (this is the
important part) *the certificate in Part II will still control the error rate anyway.*

### The code

`pact/scores.py` (and `pact_extension/scores.py`) implement `fit_probe(X, y)`:
standardize features → find the subspace the data actually spans (SVD) → fit an L2
logistic regression there. The `.score()` method returns `-p` as a NumPy array.

---

## Chapter 5. Conformal prediction from scratch

This is the mathematical heart of PACT. Read it slowly; it is simpler than it looks.

### The idea

You have a bag of **calibration** scores from *known harmful* prompts. You want to pick a
threshold τ so that a *future* harmful prompt scores **above** τ (and is therefore caught)
at least 95% of the time. Equivalently, it scores *below* τ — a **miss** — at most 5% of
the time.

Conformal prediction says: **just use one of your calibration scores as the threshold.**
Specifically, sort the harmful calibration scores from small to large and pick the k-th
smallest, where

```
k = floor((n + 1) · α)
```

with n = number of calibration scores and α = the miss rate you will tolerate (0.05).

### Why this works (the one idea to remember: exchangeability)

Suppose the future harmful prompt is *exchangeable* with the calibration ones — i.e. it
came from the same distribution, so its score is "just another draw from the same bag."
Then among the n+1 scores (n calibration + 1 new), the new one is equally likely to land
in any of the n+1 rank positions. If we set τ to the k-th smallest of the n calibration
scores, the new score falls **below** τ only if it lands in one of the bottom k positions
— probability k/(n+1). Choose k = floor((n+1)α) and that probability is ≤ α. **Done.**
No assumption about the score being good, Gaussian, calibrated — nothing. Just
exchangeability.

> This is why the guarantee held identically across all six languages in your results,
> whether the AUC was 0.80 or 0.998. Conformal control does **not** depend on the probe
> being accurate — only on cal and test being drawn the same way.

### A tiny worked example (do this by hand)

Say n = 19 harmful calibration scores and α = 0.05.
`k = floor((19+1)·0.05) = floor(1.0) = 1`.
So τ_pass = the **smallest** calibration score. A new harmful prompt misses only if it
scores below the smallest of 19 — which happens about 1/20 = 5% of the time. Exactly α.

If n = 99, `k = floor(100·0.05) = 5`, τ = the 5th smallest, miss ≈ 5/100 = 5%. As n grows
the threshold gets less noisy but the target stays 5%.

### Two flavors of the guarantee

- **Marginal** (`marginal_k`): the miss rate is ≤ α **averaged over many calibration
  sets**. This is what the six-language experiment reports.
- **PAC** (`pac_k`): stronger — for the *one* calibration set you actually drew, the miss
  rate is ≤ α with probability ≥ 1−δ. It picks the largest k such that
  `P(Binomial(n, α) ≤ k−1) ≤ δ`. Use it when you calibrate once and cannot re-draw.

### The code (exact)

```python
def marginal_k(n, alpha):            # pact/conformal.py
    return math.floor((n + 1) * alpha)

def pass_threshold(refuse_scores, alpha, delta=None):
    scores = np.sort(np.asarray(refuse_scores, float))
    k = choose_k(len(scores), alpha, delta)
    return -np.inf if k == 0 else float(scores[k - 1])   # k-th smallest, 1-indexed
```

> **A real bug worth learning from.** The original paper used `scores[k]` (the (k+1)-th
> smallest) instead of `scores[k-1]`. Off by one position → the realized miss rate was
> slightly wrong. The fix is `scores[k-1]`. And when `k == 0` (calibration set too small
> for this α), τ = −∞, so *nothing passes* and the miss rate is 0 — the safe failure mode.

---

## Chapter 6. From one threshold to a certificate (2-state, 3-state, PAC)

We now have τ_pass, which controls **misses** (harmful prompts wrongly passed). But a gate
that flags *everything* would also have zero misses and be useless. We also care about
**false flags** (harmless prompts wrongly blocked). That needs a second threshold, built
the same way but on the *harmless* calibration scores.

### The false-flag threshold

τ_flag is the mirror image of τ_pass: sort the harmless calibration scores from **large to
small**, take the k-th, with the same k-rule. A future harmless prompt scores **above**
τ_flag — a false flag — at most α_flag of the time.

```python
def flag_threshold(comply_scores, alpha, delta=None):
    scores = np.sort(np.asarray(comply_scores, float))[::-1]   # large to small
    k = choose_k(len(scores), alpha, delta)
    return np.inf if k == 0 else float(scores[k - 1])
```

### Two certificates

**Two-state (PASS / FLAG).** One threshold τ_pass.
- `s < τ_pass` → **PASS** (treat as safe). Miss rate ≤ α_miss. ✅ guaranteed.
- `s ≥ τ_pass` → **FLAG**. The false-flag rate here is *not* separately controlled.

**Three-state (PASS / DEFER / FLAG).** Two thresholds, τ_pass ≤ τ_flag.
- `s < τ_pass` → **PASS** (miss ≤ α_miss). ✅
- `s > τ_flag` → **FLAG** (false-flag ≤ α_flag). ✅
- in between → **DEFER**: "I'm not sure — send to a human / stronger check."

The three-state certificate is the honest one: it controls *both* error types, and pays
for it with an **abstention** (DEFER) region. The better your score separates the classes,
the narrower that region — so **deferral rate is a quality metric**: lower is better.

### When is three-state even possible? The AUC floor

Three-state needs τ_pass ≤ τ_flag (the PASS and FLAG regions must not overlap). If your
score is too weak, they cross and there is no valid certificate at those targets. There is
a clean necessary condition on discrimination:

```
AUC ≥ (1 − α_miss)(1 − α_flag)
```

With α_miss = 0.05 and α_flag = 0.10 this floor is `0.95 × 0.90 = 0.855`. Intuition: if a
random harmful scores above τ with prob ≥ 1−α_miss and a random harmless scores below τ
with prob ≥ 1−α_flag, then a random harmful out-ranks a random harmless (which is exactly
what AUC measures) with prob ≥ (1−α_miss)(1−α_flag). So an AUC below 0.855 *cannot* support
a two-state certificate meeting both targets — you must accept deferral. This is a
population statement: compare it against an AUC **confidence interval**, not a point
estimate.

```python
def two_state_auc_floor(alpha_miss, alpha_flag):     # pact/conformal.py
    return (1 - alpha_miss) * (1 - alpha_flag)        # 0.855 at (0.05, 0.10)
```

### How we know the code is correct

`simulate_pass_threshold` draws calibration scores as Uniform(0,1) (where the true miss
rate of a threshold τ is exactly τ) and checks empirically that the mean miss rate equals
`k/(n+1)` (marginal) and that the PAC version exceeds α at most δ of the time. This
Monte-Carlo check is `step06_verify_conformal.py`, and it passes.

---

# Part III — The PACT Method

Now we combine the probe (Ch 4) with the certificate (Ch 5–6) and solve the *cross-lingual*
problem. The certificate needs calibration scores in the target language. PACT's
contribution is **how to get a good, transportable score and how to anchor it**, giving
three variants of increasing power.

## Chapter 7. The three scores: A0, A2, TT

Let English activations be `X_en` and target-language activations `X_tgt`.

### A0 — transport (0 labels, 0 target training)

Train the probe on **English** only. Apply it, unchanged, to target activations.

```
probe = fit_probe(X_en, safe_labels_en)
s_target = probe.score(X_tgt)
```

Simplest possible transport. Works when the two languages put "refusal-worthy" along the
same direction. Weak when the target's internal geometry is rotated relative to English —
which is exactly the low-resource case.

### A2 — Procrustes alignment (0 labels, uses *parallel* prompts)

Here is the clever, label-free fix. Take **parallel** prompts (same content, English and
target). Their activations *should* mean the same thing but sit in slightly rotated
subspaces. Find the rotation that best lines the target subspace up with English, then
apply the English ruler.

Steps (all with zero labels — only paired prompts):
1. Reduce both to the top-K principal directions of English (K = 64).
2. Project both languages into that K-dim space: `P_en`, `P_tgt`.
3. Find the orthogonal rotation **R** that best maps target onto English —
   the **orthogonal Procrustes** solution: `U, _, Wᵀ = svd(P_tgtᵀ P_en)`, then `R = U Wᵀ`.
4. Rotate target into English space and score with the English probe.

```python
def fit_procrustes(X_en, X_tgt, y, k=64):     # pact_extension/scores.py
    ...
    U, _, Wh = svd(P_tgt.T @ P_en)
    rotation = U @ Wh                          # closest orthogonal map target->english
    ...
```

Why "Procrustes"? In Greek myth, Procrustes forced travelers to fit his bed. Here we
gently *rotate* (never stretch — orthogonal only, so distances are preserved) the target
space to fit the English one. A2 helps **exactly the languages where A0 is weak** — the
ones whose geometry was rotated. In your results that is Amharic, Khmer, Yoruba.

### TT — translate-train (uses translated target data)

Train the probe **directly on target-language** activations (from translated prompts). Best
discrimination, because the ruler is fit in the target's own space. The cost: you need
target data (obtained by *translating* your English set — cheap, automatic — not by
hand-labeling). This is PACT's recommended score when you can translate your calibration
set.

```
probe_tt = fit_probe(X_tgt_fit, safe_labels)
s_target = probe_tt.score(X_tgt)
```

### The ladder

| Score | Needs | Idea | Best when |
|-------|-------|------|-----------|
| **A0** | English labels only | apply English ruler as-is | languages close to English |
| **A2** | English labels + parallel prompts | rotate target onto English, then apply ruler | hard, rotated languages (am/km/yo) |
| **TT** | translated target prompts | fit ruler in target space | you can translate your set (usually) |

All three then go through the *same* conformal certificate. The score decides *quality*
(AUC, deferral); the certificate decides *safety* (miss rate). Keep those two jobs separate
in your mind — it is the cleanest way to understand PACT.

---

## Chapter 8. Parallel anchoring — why paired translations matter

The "Parallel-Anchored" in PACT is not decoration. Two places rely on prompts being
**aligned by content across languages**:

1. **A2's rotation** only makes sense if English prompt *i* and target prompt *i* are the
   *same request*. Then any difference in their activations is the *language*, not the
   *content* — which is exactly the rotation we want to remove. Parallel data isolates the
   nuisance (language) from the signal (harmfulness).

2. **Transporting the decision.** On curated benchmarks we use the gold harmful/harmless
   label as a stand-in for "what the model decides in English." In the fuller system, the
   *English* model's own refuse/comply decision is the anchor, carried to the target via
   the parallel translation. That is what makes the certificate about *the model's own
   boundary*, not about an external label set.

Translation is done offline with **NLLB-200** (a translation model), because online
translation APIs were blocked on the research server. Quality is guarded by
**back-translation** and a **language check** (Step 2 in the pipeline) so a broken
translation cannot silently poison calibration.

---

## Chapter 9. Assembling the certificate

Putting Parts II and III together, the per-language recipe is:

1. Pick a score (A0 / A2 / TT) → get scores `s` for target prompts.
2. Split the target data into **cal** and **test** (Chapter 11).
3. On **cal**: `τ_pass = pass_threshold(s of known-harmful cal, α_miss)`;
   `τ_flag = flag_threshold(s of known-harmless cal, α_flag)`.
4. On **test**: classify each prompt PASS / DEFER / FLAG with `three_state(s, τ_pass, τ_flag)`.
5. Report realized **miss** (harmful that PASSed), **false-flag** (harmless that FLAGged),
   and **deferral** rate.

The guarantee: miss ≤ α_miss and false-flag ≤ α_flag, in every language, with **no target
labels beyond the transported decision**. That is the certificate.

---

# Part IV — Evaluation

## Chapter 10. The datasets

- **AdvBench** — canonical *harmful* instructions (the "should refuse" set).
- **XSTest** — *safe prompts that look dangerous* ("how do I kill a process?"). The
  over-refusal stress test: a good gate must PASS these.
- **OR-Bench** — a larger over-refusal benchmark, same spirit as XSTest.
- **PolyRefuse** — parallel harmful/harmless prompts across **23 languages** in **3
  resource tiers**. The low-resource tier is the hard, important one:
  **sw (Swahili), am (Amharic), my (Burmese), km (Khmer), si (Sinhala), yo (Yoruba)**.
  Prompts are parallel by row index across languages — which is what makes A2's pairing
  and the decision-transport possible.

The six-language experiment (Part V) runs on PolyRefuse, reusing translated data so the
languages line up and the comparison to prior work is apples-to-apples.

## Chapter 11. Leakage-free protocol (fit / cal / test)

This is the discipline that makes the numbers honest, and it is where the biggest bug was.

**The rule:** whatever you train on, you must not evaluate on. Per language, per random
seed, split each class **40 / 30 / 40** style into three disjoint parts:

- **fit (40%)** — train TT's probe and fit A2's rotation here. A0 never touches target
  data at all (it is English-only).
- **cal (30%)** — set τ_pass and τ_flag here.
- **test (30%)** — measure AUC, miss, flag, deferral here, and *only* here.

```python
def _split(n, idx_ref, idx_com, rng, fractions=(0.4, 0.3, 0.3)):
    # each class permuted and cut by the same fractions -> disjoint fit/cal/test masks
```

> **The leakage bug (and its lesson).** An earlier version trained TT on *all* the data and
> then evaluated on the same data. Result: TT AUC = **1.000** — a perfect score, which in
> ML almost always means you are cheating without knowing it. After enforcing fit≠test, TT
> dropped to a realistic **0.978–0.998**. If you ever see 1.000, suspect leakage first.

## Chapter 12. The baseline: Aziz few-shot gate

The natural objection from a reviewer is: *"Why not just label a few target examples and
set a threshold?"* That is the **Aziz et al. few-shot** baseline, and PACT must beat it.

The few-shot gate takes k labeled examples per class from the calibration split and picks
the threshold that maximizes **macro-F1** on those few examples. It needs 2k real target
labels and gives **no guarantee** — macro-F1 optimizes balanced accuracy, not the miss
rate. We run it on the *same* TT score, so the only thing that differs is the **thresholding
rule** (conformal vs. macro-F1-on-k-labels). Apples to apples.

```python
def fewshot_threshold(harmful, harmless):   # pact_extension/fewshot_baseline.py
    # sweep candidate thresholds, return the one maximizing macro-F1 on the few labels
```

## Chapter 13. The metrics: miss, flag, deferral, AUC floor

- **miss** — fraction of harmful test prompts that PASSed (the dangerous error). Target ≤ α_miss.
- **flag** — fraction of harmless test prompts that FLAGged (the annoying error). Target ≤ α_flag.
- **defer** — fraction sent to DEFER in the 3-state certificate. Lower = better score. Not
  an error, a cost.
- **AUC** — how well the score ranks harmful above harmless. Quality only; the certificate
  controls errors regardless. Compare against the **0.855 floor** to know if a two-state
  certificate is even feasible.

---

# Part V — Results and Meaning

## Chapter 14. The six-language result, read slowly

Llama-3.1-8B, layer 10, `t_post_inst`, PolyRefuse, 30 seeds, leakage-free.

| Lang | A0 AUC | A2 AUC | TT AUC | PACT miss (A0/A2/TT) | few-shot miss k=1 | k=16 |
|------|-------:|-------:|-------:|---------------------:|------------------:|-----:|
| am   | 0.80   | 0.88   | 0.978  | ≈ 0.05               | 0.24 | 0.36 |
| sw   | 0.98   | 0.945  | 0.998  | ≈ 0.05               | 0.02 | 0.08 |
| my   | 0.89   | 0.88   | 0.983  | ≈ 0.05               | 0.06 | 0.18 |
| km   | 0.80   | 0.86   | 0.988  | ≈ 0.05               | 0.28 | 0.39 |
| si   | 0.925  | 0.90   | 0.992  | ≈ 0.05               | 0.12 | 0.18 |
| yo   | 0.82   | 0.845  | 0.987  | ≈ 0.05               | 0.24 | 0.29 |

Read it as three stories:

1. **The certificate holds everywhere.** The PACT miss column is ≈ 0.05 in *every* row,
   whether AUC is 0.80 or 0.998. This is Chapter 5 in action: conformal control does not
   care how good the score is. This is the result that *had* to be true for PACT to be
   correct — it is confirmation of the theory, not luck.
2. **TT best, A2 rescues the hard languages.** TT lifts AUC to ~0.98–0.998 everywhere
   (best discrimination, least deferral). A2 beats A0 on exactly am / km / yo — the rotated,
   low-AUC languages — and is a wash on easy Swahili. Precisely the pattern Chapter 7 predicts.
3. **Few-shot is uncontrolled and gets *worse* with more labels.** On the identical score,
   the macro-F1 gate misses 5–7× the target in the weak languages, and its miss *rises*
   with k (macro-F1 chases balanced accuracy, not miss). PACT uses zero labels and stays
   on target.

## Chapter 15. What it proves, and what it does not

**Proves:**
- A per-language miss guarantee that holds across AUC 0.80–0.998, with zero target labels.
- A concrete score ladder: TT for quality, A2 for hard languages, A0 as the floor.
- A clean, same-score win over the few-shot baseline a reviewer would propose.

**Does not (yet) prove — be honest about this:**
- These are **curated** harmful-vs-harmless prompts, so even A0 has AUC ≥ 0.80. The
  certificate was never stress-tested against a *genuinely weak* score (heavily overlapping
  classes). Theory says it will still hold there — that is the point of conformal — but you
  have not *shown* it.
- The harder **over-refusal** setting (XSTest look-alikes, where the score is weak by
  design) lives in the `safty` pipeline and has not been run at scale yet. That is the
  experiment that would *test* the guarantee rather than *confirm* it.
- One model family (Llama-3.1-8B), one layer (10). Other layers / models are open.

## Chapter 16. Where PACT sits in the literature (novelty)

PACT is best understood as a **safety-specific instance of conformal calibration transfer**
— taking a source-domain (English) decision boundary and transporting a *calibrated*,
*guaranteed* version of it to a target domain (another language) using unlabeled parallel
data. The pieces exist separately in the literature; PACT's contribution is the
*combination applied to cross-lingual LLM safety*:

- **vs. probing / representation-engineering safety work** — those transport a *direction*
  but attach no error guarantee. PACT adds the conformal certificate.
- **vs. conformal prediction papers** — those assume you have calibration labels in the
  target domain. PACT's novelty is getting them *for free* via transported decisions +
  parallel anchoring, so the target-label cost is zero.
- **vs. few-shot recalibration (Aziz-style)** — needs target labels, gives no guarantee,
  and (shown here) gets worse with more labels for the miss objective.
- **vs. Transfer/Calibration-Transfer (TCC-style) methods** — same family of ideas; PACT
  specializes them to safety, adds the A0/A2/TT ladder and the low-resource-language focus,
  and delivers a two-/three-state *actionable* gate rather than only a calibrated score.

The one-line claim: **PACT is the first to give a distribution-free, per-language miss-rate
certificate for LLM refusal that costs zero target labels, by anchoring the English
decision through parallel translations and calibrating conformally.**

---

# Part VI — The Code

## Chapter 17. The `safty` pipeline in 8 steps

A clean, modular reimplementation. Each step is one script under `scripts/`, reading and
writing cached artifacts so you can stop and resume.

| Step | Script | Does |
|------|--------|------|
| 1 | `step01_build_datasets.py` | assemble AdvBench / XSTest / OR-Bench / PolyRefuse into a common schema |
| 2 | `step02_translate.py` | NLLB translation + back-translation + language guard |
| 3 | `step03_generate.py` | run the model to get refuse/comply behavior |
| 4 | `step04_label.py` | label refuse vs comply (the decision to certify) |
| 5 | `step05_extract_features.py` | cache layer-10 `t_post_inst` activations |
| 6 | `step06_verify_conformal.py` | Monte-Carlo proof the conformal code is correct |
| 7 | `step07_contrast_experiment.py` | A0/A2/TT + certificate on the contrast (refusal vs topic) setting |
| 8 | `step08_base_experiment.py` | the base PACT tables |

The library under `pact/` holds the reusable pieces: `conformal.py` (Ch 5–6),
`scores.py` (Ch 4, 7), `evaluation.py` (Ch 13), `experiments.py`, `report.py`, plus
`config.py` / `paths.py` / `io.py` for plumbing and `data/` for dataset construction.
35 unit tests cover it.

## Chapter 18. The `pact_extension` on real activations

Living in the forked `low-resource-safety` repo, this runs the six-language study on the
original authors' cached activations, next to their few-shot baseline:

- `scores.py` — A0 / A2 / TT probes (Ch 7).
- `conformal.py` — the certificate (copied from `safty`, Ch 5–6).
- `fewshot_baseline.py` — the Aziz gate (Ch 12).
- `run_pact_polyrefuse.py` — the full leakage-free evaluation (Ch 11), writes
  `results/pact_v3.csv` and `pact_v3_summary.csv`.
- `docs/results_pact_polyrefuse.md` — the write-up of Chapter 14.

## Chapter 19. Running everything

**The safty pipeline** (each step reads the previous step's output):

```bash
python scripts/step01_build_datasets.py
python scripts/step02_translate.py
python scripts/step03_generate.py
python scripts/step04_label.py
python scripts/step05_extract_features.py
python scripts/step06_verify_conformal.py     # proves the math; needs no GPU
python scripts/step07_contrast_experiment.py
python scripts/step08_base_experiment.py
```

**The six-language extension** (on a machine with the activation cache):

```bash
python pact_extension/run_pact_polyrefuse.py \
    --activations-root artifacts/activations/llama --layer 10 \
    --target-langs am sw my km si yo --seeds 30 \
    --out pact_extension/results/pact_v3.csv
```

Notes learned the hard way on the research server: use the env that actually has the deps
(`scipy`, `hydra`); for long-token languages (Burmese) drop `extraction.batch_size` and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid CUDA OOM; translation is offline
via NLLB because online APIs are blocked.

---

# Appendix A — Glossary (plain words)

- **Activation** — the model's internal vector for a prompt at one layer/position.
- **Probe** — a simple linear ruler over activations that scores "how harmful."
- **Score `s`** — `-P(safe)`; high = looks harmful. One consistent orientation everywhere.
- **Calibration (cal)** — a held-out set used *only* to set thresholds, never to train or test.
- **Conformal prediction** — a way to turn any score into a threshold with a provable error
  rate, needing only exchangeability.
- **Exchangeability** — cal and test items are drawn the same way, so any one is "just
  another draw." The single assumption behind the guarantee.
- **Miss** — a harmful prompt wrongly PASSed. The dangerous error. Controlled at α_miss.
- **False flag** — a harmless prompt wrongly FLAGged. The annoying error. Controlled at α_flag.
- **Deferral** — the DEFER (abstain) region of the 3-state certificate. A cost, not an error.
- **A0 / A2 / TT** — transport / Procrustes-aligned / translate-train scores (Ch 7).
- **Procrustes** — the label-free rotation that lines the target space up with English.
- **Certificate** — the whole gate + its guaranteed error bound.
- **Marginal vs PAC** — guarantee-on-average vs guarantee-for-this-one-calibration-set.

# Appendix B — Symbols

| Symbol | Meaning |
|--------|---------|
| **x** | activation vector for a prompt |
| **w**, *b* | probe weight vector, bias |
| *s* | score, `-P(safe)`, high = harmful |
| *n* | number of calibration scores |
| α_miss, α_flag | tolerated miss / false-flag rates (0.05, 0.10) |
| δ | PAC failure probability |
| *k* | order-statistic index, `floor((n+1)α)` (marginal) |
| τ_pass, τ_flag | pass / flag thresholds |
| K | Procrustes principal components (64) |
| **R** | Procrustes rotation matrix |

# Appendix C — FAQ

**Q: If the probe is weak (low AUC), isn't the certificate meaningless?**
No — the *miss guarantee* still holds (Ch 5). What a weak probe costs you is more
**deferral** in the 3-state certificate, not more misses. Safety is preserved; efficiency
degrades gracefully.

**Q: Why not just fine-tune the model to be safe in each language?**
Expensive, needs labels in every language, and still gives no per-language guarantee. PACT
is a *lightweight, label-free, guaranteed* gate you put around any model.

**Q: What is the single most important sentence in the whole method?**
"The score decides quality; the certificate decides safety — and the certificate's
guarantee does not depend on the score being good." Everything else is machinery around
that.

**Q: Where is the guarantee weakest?**
In the *exchangeability* assumption. If the target-language test prompts are drawn
differently from the calibration prompts (a distribution shift *within* the language), the
bound can slip. Parallel anchoring and honest cal/test splitting are what keep
exchangeability believable.

---

*End of tutorial. For the honest results write-up see
`pact_extension/docs/results_pact_polyrefuse.md`; for the conformal code that every
guarantee rests on see `pact/conformal.py`.*
