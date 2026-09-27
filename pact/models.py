"""Loading the language models, formatting prompts, generating, and reading hidden states.

Steps 3, 4 and 5 all go through `encode`, so the model sees exactly the same
prompt text when it answers (step 3) as when we read its hidden states (step 5).
"""
from __future__ import annotations

import numpy as np


def load_model(model_cfg: dict):
    """Returns (tokenizer, model). `model_cfg` is one entry of config['models']."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    extra = {"gguf_file": model_cfg["gguf_file"]} if model_cfg.get("gguf_file") else {}
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], **extra)
    # transformers >= 4.56 calls the argument `dtype`; older versions call it `torch_dtype`.
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"], device_map="auto", **{dtype_key: torch.bfloat16}, **extra
    )
    model.eval()
    return prepare_tokenizer(tokenizer), model


def prepare_tokenizer(tokenizer):
    """Pad on the left, so the last position of every row is the prompt's last token."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_prompt(tokenizer, text: str, style: str, template: str = "{prompt}\n") -> str:
    """'chat' uses the model's own chat template; 'completion' is plain text (base model)."""
    if style == "chat":
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if style == "completion":
        return template.format(prompt=text)
    raise ValueError(f"unknown prompt style {style!r}")


def encode(tokenizer, model, texts: list[str], style: str, template: str):
    """Format + tokenize a batch and move it to the model's device.

    The chat template already contains the begin-of-text token, so special tokens are
    only added for the completion style (otherwise the BOS token would appear twice).
    """
    prompts = [format_prompt(tokenizer, t, style, template) for t in texts]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=(style == "completion"))
    return batch.to(model.device)


def generate(tokenizer, model, texts: list[str], style: str, template: str, max_new_tokens: int) -> list[str]:
    """Greedy decoding. Returns only the newly generated text for each prompt."""
    import torch

    batch = encode(tokenizer, model, texts, style, template)
    with torch.no_grad():
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[:, batch["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def last_token_states(tokenizer, model, texts: list[str], style: str, template: str,
                      layers: list[int]) -> dict[int, np.ndarray]:
    """Hidden state of the last prompt token at each requested layer: {layer: (n, d) array}.

    hidden_states[0] is the embedding output and hidden_states[L] the output of layer L.
    With left padding, position ids must be computed from the attention mask; otherwise
    padded rows would get shifted positions and different hidden states.
    """
    import torch

    batch = encode(tokenizer, model, texts, style, template)
    position_ids = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(**batch, position_ids=position_ids, output_hidden_states=True)
    return {layer: out.hidden_states[layer][:, -1, :].float().cpu().numpy() for layer in layers}
