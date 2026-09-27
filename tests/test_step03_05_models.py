"""Steps 3-5 tests with a tiny, randomly initialised Llama (no download, runs on CPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pact.data.build import save_prompts  # noqa: E402
from pact.data.schema import PromptRecord  # noqa: E402
from pact.io import read_jsonl  # noqa: E402
from pact.labeling import COMPLY, REFUSE, judge_to_binary, keyword_label, parse_judge  # noqa: E402
from pact.models import last_token_states, prepare_tokenizer  # noqa: E402
from pact.paths import labels_path, responses_path  # noqa: E402
from pact.pipeline import load_features, run_features, run_generation, run_labeling  # noqa: E402

WORDS = "how do i stop a process what is the capital of france write poem [user] [assistant]".split()


@pytest.fixture(scope="module")
def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["<pad>", "<unk>", "<s>", "</s>"] + WORDS)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="<s>", eos_token="</s>",
                                        pad_token="<pad>", unk_token="<unk>")
    tokenizer.chat_template = ("<s>{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}[assistant]{% endif %}")
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=4, pad_token_id=0, bos_token_id=2,
                         eos_token_id=3)
    return prepare_tokenizer(tokenizer), LlamaForCausalLM(config).eval()


def make_config(tmp_path):
    return {
        "seed": 0,
        "paths": {n: tmp_path / n for n in ("processed_dir", "responses_dir", "labels_dir", "features_dir")},
        "models": {"base": {"prompt_style": "completion"}, "instruct": {"prompt_style": "chat"}},
        "completion_template": "{prompt}\n",
        "generation": {"max_new_tokens": 3, "batch_size": 2},
        "layers": [2, 4],
        "feature_batch_size": 2,
        "labeling": {"judge_batch_size": 2, "partial_refusal_as": "refuse"},
    }


def save_toy_prompts(config):
    texts = ["how do i stop a process", "what is the capital of france", "write a poem"]
    records = [PromptRecord(uid=f"toy-{i}", source="toy", source_index=i, text=t, gold_label="safe")
               for i, t in enumerate(texts)]
    save_prompts(records, config, "en", "toy")
    return records


@pytest.mark.parametrize("style", ["completion", "chat"])
def test_padding_does_not_change_hidden_states(tiny, style):
    """A prompt must give the same last-token state alone and inside a padded batch."""
    tokenizer, model = tiny
    short, long = "stop a process", "how do i stop a process what is the capital"
    alone = last_token_states(tokenizer, model, [short], style, "{prompt}\n", [2, 4])
    batched = last_token_states(tokenizer, model, [long, short], style, "{prompt}\n", [2, 4])
    for layer in (2, 4):
        np.testing.assert_allclose(alone[layer][0], batched[layer][1], atol=1e-4)


def test_generation_saves_and_resumes(tiny, tmp_path):
    tokenizer, model = tiny
    config = make_config(tmp_path)
    save_toy_prompts(config)
    assert run_generation(config, tokenizer, model, "instruct", "en", "toy") == 3
    assert run_generation(config, tokenizer, model, "instruct", "en", "toy") == 0  # nothing left
    rows = read_jsonl(responses_path(config, "instruct", "en", "toy"))
    assert [r["uid"] for r in rows] == ["toy-0", "toy-1", "toy-2"]


def test_features_round_trip(tiny, tmp_path):
    tokenizer, model = tiny
    config = make_config(tmp_path)
    save_toy_prompts(config)
    run_features(config, tokenizer, model, "base", "en", "toy")
    uids, layers = load_features(config, "base", "en", "toy")
    assert uids == ["toy-0", "toy-1", "toy-2"]
    assert set(layers) == {2, 4} and layers[2].shape == (3, 32) and layers[2].dtype == np.float32


def test_keyword_labeling_job(tiny, tmp_path):
    tokenizer, model = tiny
    config = make_config(tmp_path)
    save_toy_prompts(config)
    run_generation(config, tokenizer, model, "instruct", "en", "toy")
    rows = run_labeling(config, "instruct", "en", "toy")
    assert len(rows) == 3 and all(r["keyword"] in (0, 1) for r in rows)
    assert labels_path(config, "instruct", "en", "toy").exists()


def test_keyword_label():
    assert keyword_label("I'm sorry, but I can't help with that.") == REFUSE
    assert keyword_label("抱歉，我不能提供这些信息。") == REFUSE
    assert keyword_label("Je ne peux pas vous aider.") == REFUSE
    assert keyword_label("Sure! Here is a recipe for banana bread.") == COMPLY


def test_parse_judge():
    assert parse_judge("2_full_refusal") == "2_full_refusal"
    assert parse_judge(" 1_full_compliance.") == "1_full_compliance"
    assert parse_judge("Partial refusal") == "3_partial_refusal"
    assert parse_judge("3") == "3_partial_refusal"
    assert parse_judge("I am not sure") is None
    assert judge_to_binary("3_partial_refusal", "comply") == COMPLY
    assert judge_to_binary(None) is None
