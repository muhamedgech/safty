"""Machine translation with NLLB, plus checks that the output really is the target language."""
from __future__ import annotations

# Unicode blocks used to confirm the output is written in the expected script.
SCRIPT_RANGES = {
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],  # CJK ideographs
    "am": [(0x1200, 0x139F), (0x2D80, 0x2DDF)],  # Ethiopic
}


class Translator:
    """Loads the NLLB model once and translates batches of text between any two languages."""

    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def __call__(self, texts: list[str], src: str, tgt: str, batch_size: int = 16,
                 max_new_tokens: int = 100) -> list[str]:
        import torch

        self.tokenizer.src_lang = src
        target_token = self.tokenizer.convert_tokens_to_ids(tgt)
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=target_token,  # tells NLLB which language to write
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,                   # greedy, as in the paper
                )
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
            print(f"  translated {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return [o.strip() for o in outputs]


def script_fraction(text: str, lang: str) -> float:
    """Share of the letters in `text` that belong to `lang`'s script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ranges = SCRIPT_RANGES[lang]
    return sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters) / len(letters)


def check_translation(sources: list[str], outputs: list[str], lang: str,
                      max_identical: float = 0.05, min_in_script: float = 0.9) -> dict:
    """Stop the pipeline if a 'translation' is empty, just English, or in the wrong script.

    This is the guard that would have caught the paper's Appendix A bug, where French
    and Amharic files silently contained the English prompts.
    """
    if len(sources) != len(outputs):
        raise ValueError(f"{lang}: {len(sources)} inputs but {len(outputs)} outputs")
    empty = sum(not o.strip() for o in outputs)
    identical = sum(s.strip() == o.strip() for s, o in zip(sources, outputs))
    report = {"n": len(outputs), "empty": empty, "identical_to_source": identical}
    if empty:
        raise ValueError(f"{lang}: {empty} empty translations")
    if identical > max_identical * len(outputs):
        raise ValueError(f"{lang}: {identical}/{len(outputs)} outputs are identical to the English input")
    if lang in SCRIPT_RANGES:
        in_script = sum(script_fraction(o, lang) >= 0.5 for o in outputs)
        report["in_target_script"] = in_script
        if in_script < min_in_script * len(outputs):
            raise ValueError(f"{lang}: only {in_script}/{len(outputs)} outputs are in the {lang} script")
    return report
