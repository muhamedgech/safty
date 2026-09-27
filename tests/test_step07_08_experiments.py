"""Steps 7-8 tests: probes, splits, metrics, and both experiment scripts end to end on synthetic data."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from pact.config import load_config
from pact.data.build import save_prompts
from pact.data.schema import PromptRecord
from pact.evaluation import CAL, FIT, TEST, auc, certify, corrected_ci, stratified_partition
from pact.experiments import load_contrast_data, shifted_sample
from pact.io import write_jsonl
from pact.paths import features_path, labels_path
from pact.scores import fit_probe, fit_procrustes_probe

ROOT = Path(__file__).resolve().parent.parent
LAYERS = [8, 16]
D = 24


# ------------------------------------------------------------------ unit tests

def test_partition_keeps_strata_proportions():
    strata = ["a"] * 100 + ["b"] * 50
    split = stratified_partition(strata, (0.4, 0.3, 0.3), seed=0)
    assert (split[:100] == FIT).sum() == 40 and (split[100:] == CAL).sum() == 15
    assert (split == stratified_partition(strata, (0.4, 0.3, 0.3), seed=0)).all()


def test_probe_finds_the_informative_layer():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    feats = {8: rng.standard_normal((200, D)), 16: rng.standard_normal((200, D))}
    feats[16][:, 0] += 2 * y
    probe = fit_probe(feats, y, [0.01, 0.1, 1.0])
    assert probe.layer == 16
    assert auc(probe.score(feats[16]), y == 0) > 0.85  # high score = refusal-worthy


def test_procrustes_undoes_a_rotation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    X_en = rng.standard_normal((300, D))
    X_en[:, 0] += 2 * y
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    X_tgt = X_en @ Q + 0.05 * rng.standard_normal((300, D))  # target = rotated English
    a0 = fit_probe({0: X_en}, y, [0.1, 1.0])
    a2 = fit_procrustes_probe(X_en, X_tgt, y, k=D, Cs=[0.1, 1.0])
    assert auc(a2.score(X_tgt), y == 0) > 0.9 > auc(a0.score(X_tgt), y == 0) + 0.1


def test_certify_marginal_miss_rate_is_near_alpha():
    rng = np.random.default_rng(1)
    misses = []
    for seed in range(300):
        z = rng.integers(0, 2, 600)
        s = rng.standard_normal(600) - 1.5 * z
        split = stratified_partition(list(z), (0.4, 0.3, 0.3), seed)
        row = certify(s, z, 1 - z, split == CAL, split == TEST, 0.05, 0.10, None)
        misses.append(row["miss_2state"])
    assert np.mean(misses) == pytest.approx(0.05, abs=0.01)


def test_corrected_ci_is_wider_than_naive():
    values = np.random.default_rng(0).normal(0.8, 0.02, 50)
    mean, lo, hi = corrected_ci(values, n_train=240, n_test=180)
    naive = 1.96 * values.std(ddof=1) / np.sqrt(50)
    assert hi - lo > 4 * naive


def test_shifted_sample_hits_target_rate():
    z = np.array([1] * 30 + [0] * 90)
    chosen = shifted_sample(np.arange(120), z, 0.10, np.random.default_rng(0))
    assert z[chosen].mean() == pytest.approx(0.10, abs=0.01)
    assert len(set(chosen)) == len(chosen)  # without replacement


# ------------------------------------------------------------------ end to end

def write_world(tmp_path: Path) -> Path:
    """A synthetic project: prompts, labels and features for every file steps 7-8 read."""
    rng = np.random.default_rng(0)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["paths"] = {k: str(tmp_path / k) for k in cfg["paths"]}
    cfg["layers"] = LAYERS
    cfg["probe"].update(Cs=[0.1, 1.0], procrustes_k=8)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    config = load_config(cfg_path)
    rotations = {lang: np.linalg.qr(rng.standard_normal((D, D)))[0] for lang in ("zh", "fr", "am")}

    def states(z, lang, topic=None):
        X = rng.standard_normal((len(z), D))
        X[:, 0] += 2.0 * (1 - np.asarray(z))           # refusal direction
        if topic is not None:
            X[:, 1] += 2.0 * np.asarray(topic)          # topic direction
        if lang in rotations:
            X = 0.7 * X + 0.3 * X @ rotations[lang]     # partially different geometry
        return {f"layer_{L}": X + 0.1 * rng.standard_normal(X.shape) for L in LAYERS}

    def save(model, source, records, z, langs, topic=None):
        save_prompts(records, config, "en", source)
        write_jsonl([{"uid": r.uid, "keyword": int(v), "judge": int(v), "judge_class": None}
                     for r, v in zip(records, z)], labels_path(config, model, "en", source))
        for lang in langs:
            path = features_path(config, model, lang, source)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, uids=np.array([r.uid for r in records]), **states(z, lang, topic))

    langs = ["en", "zh", "fr", "am"]
    # contrast set: 150 "xstest" + 50 "orbench"
    for source, n in (("xstest", 150), ("orbench_hard", 50)):
        gold = rng.integers(0, 2, n) if source == "xstest" else np.zeros(n, int)
        z = np.where(rng.random(n) < 0.85, 1 - gold, gold)  # the model mostly follows gold
        recs = [PromptRecord(f"{source}-{i:04d}", source, i, f"{source} prompt {i}",
                             "unsafe" if g else "safe", translate=True) for i, g in enumerate(gold)]
        save("instruct", source, recs, z, langs, topic=gold)
    # topic-control training data and the base experiment's AdvBench
    adv = [PromptRecord(f"advbench-{i:04d}", "advbench", i, f"harmful {i}", "unsafe", translate=i < 120)
           for i in range(200)]
    save("instruct", "advbench", adv, np.zeros(200, int), ["en"], topic=np.ones(200))
    ben = [PromptRecord(f"benign-{i:04d}", "benign", i, f"benign {i}", "safe") for i in range(60)]
    save("instruct", "benign", ben, np.ones(60, int), ["en"], topic=np.zeros(60))
    z_adv = (rng.random(200) < 0.3).astype(int)
    save_prompts(adv, config, "en", "advbench")
    write_jsonl([{"uid": r.uid, "keyword": int(v), "judge": int(v)} for r, v in zip(adv, z_adv)],
                labels_path(config, "base", "en", "advbench"))
    for lang in langs:
        path = features_path(config, "base", lang, "advbench")
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = [r for r in adv if r.translate or lang == "en"]
        zk = [z_adv[r.source_index] for r in keep]
        np.savez(path, uids=np.array([r.uid for r in keep]), **states(zk, lang))
    return cfg_path


def test_contrast_data_loads_and_aligns(tmp_path):
    config = load_config(write_world(tmp_path))
    data = load_contrast_data(config)
    assert len(data.uids) == 200 and data.feats["am"][8].shape == (200, D)
    assert data.topic_y.sum() == 60


@pytest.mark.parametrize("script", ["step07_contrast_experiment.py", "step08_base_experiment.py"])
def test_experiment_scripts_run(tmp_path, script):
    cfg_path = write_world(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--config", str(cfg_path),
                             "--partitions", "3", "--n-jobs", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    out = next((tmp_path / "results_dir").iterdir())
    assert (out / "partitions.csv").exists() and (out / "summary.csv").exists()
