"""The work done by steps 3, 4 and 5 for one (model, language, source) job.

Scripts only choose which jobs to run and load the model; everything here is
testable with a tiny model.
"""
from __future__ import annotations

import numpy as np

from pact.data.build import load_source
from pact.io import append_jsonl, read_jsonl, write_jsonl
from pact.labeling import judge_prompt, judge_to_binary, keyword_label, parse_judge
from pact.models import generate, last_token_states
from pact.paths import features_path, labels_path, responses_path


def iter_jobs(job_table: dict, models: list[str] | None = None):
    """Yield (model, lang, source) from a config table like config['generate']."""
    for model, sources in job_table.items():
        if models and model not in models:
            continue
        for source, langs in sources.items():
            for lang in langs:
                yield model, lang, source


def run_generation(config: dict, tokenizer, model, model_key: str, lang: str, source: str) -> int:
    """Step 3. Answers every prompt; progress is saved after each batch, so it can resume."""
    path = responses_path(config, model_key, lang, source)
    done = {r["uid"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in load_source(source, config, lang) if r.uid not in done]
    style = config["models"][model_key]["prompt_style"]
    gcfg = config["generation"]
    for start in range(0, len(todo), gcfg["batch_size"]):
        batch = todo[start:start + gcfg["batch_size"]]
        answers = generate(tokenizer, model, [r.text for r in batch], style,
                           config["completion_template"], gcfg["max_new_tokens"])
        append_jsonl(({"uid": r.uid, "prompt": r.text, "response": a} for r, a in zip(batch, answers)), path)
        print(f"  {model_key}/{lang}/{source}: {len(done) + start + len(batch)}/{len(done) + len(todo)}", end="\r")
    print()
    return len(todo)


def run_labeling(config: dict, model_key: str, lang: str, source: str, judge=None) -> list[dict]:
    """Step 4. Keyword label always; judge label when `judge` = (tokenizer, model) is given.

    The judge sees the English original of the question (the response may be in any language).
    """
    responses = read_jsonl(responses_path(config, model_key, lang, source))
    english = {r.uid: r.text for r in load_source(source, config, "en")}
    rows = [{"uid": r["uid"], "keyword": keyword_label(r["response"]), "judge": None, "judge_class": None}
            for r in responses]
    if judge is not None:
        tokenizer, model = judge
        lcfg = config["labeling"]
        size = lcfg["judge_batch_size"]
        for start in range(0, len(responses), size):
            batch = responses[start:start + size]
            prompts = [judge_prompt(english.get(r["uid"], r["prompt"]), r["response"]) for r in batch]
            outputs = generate(tokenizer, model, prompts, "chat", "", max_new_tokens=16)
            for row, out in zip(rows[start:start + size], outputs):
                row["judge_class"] = parse_judge(out)
                row["judge"] = judge_to_binary(row["judge_class"], lcfg["partial_refusal_as"])
            print(f"  judged {min(start + size, len(rows))}/{len(rows)}", end="\r")
        print()
    write_jsonl(rows, labels_path(config, model_key, lang, source))
    return rows


def run_features(config: dict, tokenizer, model, model_key: str, lang: str, source: str) -> None:
    """Step 5. Saves the last-prompt-token hidden state at every configured layer."""
    records = load_source(source, config, lang)
    style = config["models"][model_key]["prompt_style"]
    layers = config["layers"]
    size = config["feature_batch_size"]
    chunks = {layer: [] for layer in layers}
    for start in range(0, len(records), size):
        batch = records[start:start + size]
        states = last_token_states(tokenizer, model, [r.text for r in batch], style,
                                   config["completion_template"], layers)
        for layer in layers:
            chunks[layer].append(states[layer].astype(np.float16))  # half size on disk
        print(f"  {model_key}/{lang}/{source}: {min(start + size, len(records))}/{len(records)}", end="\r")
    print()
    path = features_path(config, model_key, lang, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"layer_{layer}": np.concatenate(chunks[layer]) for layer in layers}
    np.savez(path, uids=np.array([r.uid for r in records]), **arrays)


def load_features(config: dict, model_key: str, lang: str, source: str) -> tuple[list[str], dict[int, np.ndarray]]:
    """(uids, {layer: float32 array}) as saved by step 5."""
    with np.load(features_path(config, model_key, lang, source)) as data:
        uids = [str(u) for u in data["uids"]]
        layers = {int(k.split("_")[1]): data[k].astype(np.float32) for k in data.files if k.startswith("layer_")}
    return uids, layers
