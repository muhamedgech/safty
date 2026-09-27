"""The two experiments of the paper.

contrast (step 7)  instruct model, XSTest + OR-Bench-Hard: topic control vs PACT-A0/A2/TT
                   (Tables 2', 3', 4', 2'', 2''')
base     (step 8)  base model, AdvBench: PACT vs target-native, plus the label-shift test
                   (Tables 1, 1b)

Each experiment has a loader (reads steps 1-5 from disk into plain arrays) and a
per-partition function (pure computation, tested with synthetic data).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pact.data.build import load_source
from pact.evaluation import CAL, FIT, TEST, certify, stratified_partition
from pact.labeling import load_labels
from pact.paths import labels_path
from pact.pipeline import load_features
from pact.scores import fit_probe, fit_procrustes_probe


def align(wanted: list[str], uids: list[str], feats: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Reorder feature rows to follow `wanted`. Fails loudly if a prompt is missing."""
    index = {u: i for i, u in enumerate(uids)}
    missing = [u for u in wanted if u not in index]
    if missing:
        raise KeyError(f"{len(missing)} prompts have no features, e.g. {missing[:3]}")
    rows = [index[u] for u in wanted]
    return {layer: X[rows] for layer, X in feats.items()}


def regimes(config: dict) -> list[tuple[str, float | None]]:
    return [("marginal", None)] + [("pac", d) for d in config["conformal"]["deltas"]]


# ------------------------------------------------------------------ contrast experiment

@dataclass
class ContrastData:
    uids: list[str]
    z: np.ndarray                     # English model's decision: 1 comply / 0 refuse
    gold_refuse: np.ndarray           # 1 if the dataset marks the prompt unsafe
    strata: list[str]                 # source/gold/z, for stratified splits
    feats: dict[str, dict[int, np.ndarray]]   # lang -> layer -> (n, d), rows follow uids
    topic_feats: dict[int, np.ndarray]        # English AdvBench + benign prompts
    topic_y: np.ndarray                       # 1 benign / 0 harmful
    stable: dict[str, np.ndarray] = field(default_factory=dict)  # lang -> round-trip-stable mask


def load_contrast_data(config: dict, label_method: str | None = None) -> ContrastData:
    ecfg = config["contrast_experiment"]
    model, method = ecfg["model"], label_method or ecfg["label_method"]
    uids, z, gold, strata = [], [], [], []
    for source in ecfg["sources"]:
        labels = load_labels(config, model, "en", source, method)
        records = [r for r in load_source(source, config) if r.translate]
        dropped = [r.uid for r in records if r.uid not in labels]
        if dropped:
            print(f"  {source}: {len(dropped)} prompts without a {method} label are left out")
        for r in records:
            if r.uid in labels:
                uids.append(r.uid)
                z.append(labels[r.uid])
                gold.append(int(r.gold_label == "unsafe"))
                strata.append(f"{source}/{r.gold_label}/{labels[r.uid]}")

    feats = {}
    for lang in ["en"] + ecfg["target_langs"]:
        per_source = [load_features(config, model, lang, s) for s in ecfg["sources"]]
        all_uids = [u for u_list, _ in per_source for u in u_list]
        stacked = {L: np.concatenate([f[L] for _, f in per_source]) for L in per_source[0][1]}
        feats[lang] = align(uids, all_uids, stacked)

    harm_uids, harm = load_features(config, model, "en", "advbench")
    safe_uids, safe = load_features(config, model, "en", "benign")
    topic_feats = {L: np.concatenate([harm[L], safe[L]]) for L in harm}
    topic_y = np.array([0] * len(harm_uids) + [1] * len(safe_uids))

    stable = {}
    z_by_uid = dict(zip(uids, z))
    for lang in ecfg["target_langs"]:
        back = {}
        for source in ecfg["sources"]:
            if labels_path(config, model, f"{lang}-back", source).exists():
                back.update(load_labels(config, model, f"{lang}-back", source, method))
        if back:  # prompts without a round trip (e.g. OR-Bench) count as not verified stable
            stable[lang] = np.array([u in back and back[u] == z_by_uid[u] for u in uids])

    return ContrastData(uids, np.array(z), np.array(gold), strata, feats, topic_feats, topic_y, stable)


def fit_topic_probe(data: ContrastData, config: dict):
    """Trained once: it never sees the contrast set, only its threshold is recalibrated."""
    pcfg = config["probe"]
    return fit_probe(data.topic_feats, data.topic_y, pcfg["Cs"], pcfg["cv_folds"], config["seed"])


def contrast_scores(data: ContrastData, split: np.ndarray, topic, config: dict, seed: int) -> dict:
    """{(method, lang): scores for all prompts}. Every probe is fitted on the fit split only."""
    pcfg = config["probe"]
    fit = split == FIT
    zf = data.z[fit]
    en_fit = {L: X[fit] for L, X in data.feats["en"].items()}
    a0 = fit_probe(en_fit, zf, pcfg["Cs"], pcfg["cv_folds"], seed)
    scores = {("english", "en"): a0.score_layers(data.feats["en"])}
    for lang in config["contrast_experiment"]["target_langs"]:
        X = data.feats[lang]
        scores[("topic", lang)] = topic.score_layers(X)
        scores[("a0", lang)] = a0.score_layers(X)
        a2 = fit_procrustes_probe(en_fit[a0.layer], X[a0.layer][fit], zf, pcfg["procrustes_k"],
                                  pcfg["Cs"], pcfg["cv_folds"], seed)
        scores[("a2", lang)] = a2.score(X[a0.layer])
        tt = fit_probe({L: A[fit] for L, A in X.items()}, zf, pcfg["Cs"], pcfg["cv_folds"], seed)
        scores[("tt", lang)] = tt.score_layers(X)
    return scores


def run_contrast_partition(data: ContrastData, topic, config: dict, partition: int,
                           stable_only: bool = False) -> list[dict]:
    seed = config["seed"] + partition
    split = stratified_partition(data.strata, tuple(config["split"]["fractions"]), seed)
    ccfg = config["conformal"]
    rows = []
    for (method, lang), s in contrast_scores(data, split, topic, config, seed).items():
        keep = data.stable[lang] if stable_only and lang in data.stable else np.ones(len(s), bool)
        for regime, delta in regimes(config):
            row = certify(s, data.z, data.gold_refuse, (split == CAL) & keep, (split == TEST) & keep,
                          ccfg["alpha_miss"], ccfg["alpha_flag"], delta,
                          config["contrast_experiment"]["frontier_alphas"] if delta is None else ())
            rows.append({"partition": partition, "method": method, "lang": lang,
                         "regime": regime, "delta": delta, **row})
    return rows


# ------------------------------------------------------------------ base experiment

@dataclass
class BaseData:
    shared_uids: list[str]            # translated prompts (calibration/test come only from these)
    z: np.ndarray                     # English decision on the shared prompts
    feats: dict[str, dict[int, np.ndarray]]   # lang -> layer -> states of the shared prompts
    extra_en_feats: dict[int, np.ndarray]     # English-only prompts, used only for fitting
    extra_z: np.ndarray


def load_base_data(config: dict, label_method: str | None = None) -> BaseData:
    ecfg = config["base_experiment"]
    model, source = ecfg["model"], ecfg["source"]
    labels = load_labels(config, model, "en", source, label_method or ecfg["label_method"])
    records = [r for r in load_source(source, config) if r.uid in labels]
    shared = [r.uid for r in records if r.translate]
    extra = [r.uid for r in records if not r.translate]
    en_uids, en_feats = load_features(config, model, "en", source)
    feats = {"en": align(shared, en_uids, en_feats)}
    for lang in ecfg["target_langs"]:
        feats[lang] = align(shared, *load_features(config, model, lang, source))
    return BaseData(shared, np.array([labels[u] for u in shared]), feats,
                    align(extra, en_uids, en_feats), np.array([labels[u] for u in extra]))


def run_base_partition(data: BaseData, config: dict, partition: int) -> list[dict]:
    """PACT (English-fit probe) vs target-native (TT) on the shared prompts, plus label shift."""
    seed = config["seed"] + partition
    ecfg, pcfg, ccfg = config["base_experiment"], config["probe"], config["conformal"]
    split = stratified_partition(list(data.z), tuple(config["split"]["fractions"]), seed)
    fit = split == FIT
    en_fit = {L: np.concatenate([X[fit], data.extra_en_feats[L]]) for L, X in data.feats["en"].items()}
    pact = fit_probe(en_fit, np.concatenate([data.z[fit], data.extra_z]), pcfg["Cs"], pcfg["cv_folds"], seed)
    gold = np.ones(len(data.z), int)  # every AdvBench prompt is unsafe; auc_gold is undefined here
    rows = []
    for lang in ecfg["target_langs"]:
        X = data.feats[lang]
        native = fit_probe({L: A[fit] for L, A in X.items()}, data.z[fit], pcfg["Cs"], pcfg["cv_folds"], seed)
        for method, s in (("pact", pact.score_layers(X)), ("native", native.score_layers(X))):
            row = certify(s, data.z, gold, split == CAL, split == TEST, ccfg["alpha_miss"], ccfg["alpha_flag"], None)
            rows.append({"partition": partition, "lang": lang, "method": method, "pi": None, **row})

        # Label shift: rebuild calibration + test from the non-fit pool with compliance rate pi.
        s = pact.score_layers(X)
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(~fit)
        for pi in ecfg["label_shift_pis"]:
            chosen = shifted_sample(pool, data.z, pi, rng)
            if chosen is None:
                continue
            half = stratified_partition(list(data.z[chosen]), (0.0, 0.5, 0.5), seed)
            cal = np.zeros(len(s), bool)
            test = np.zeros(len(s), bool)
            cal[chosen[half == CAL]] = True
            test[chosen[half == TEST]] = True
            row = certify(s, data.z, gold, cal, test, ccfg["alpha_miss"], ccfg["alpha_flag"], None)
            rows.append({"partition": partition, "lang": lang, "method": "pact", "pi": pi, **row})
    return rows


def shifted_sample(pool: np.ndarray, z: np.ndarray, pi: float, rng) -> np.ndarray | None:
    """Largest subset of `pool` (without replacement) whose compliance rate is pi."""
    comply, refuse = pool[z[pool] == 1], pool[z[pool] == 0]
    if len(comply) / len(pool) > pi:
        n_comply, n_refuse = int(pi * len(refuse) / (1 - pi)), len(refuse)
    else:
        n_comply, n_refuse = len(comply), int(len(comply) * (1 - pi) / pi)
    if n_comply < 2 or n_refuse < 2:
        return None
    return np.concatenate([rng.choice(comply, n_comply, replace=False), rng.choice(refuse, n_refuse, replace=False)])
