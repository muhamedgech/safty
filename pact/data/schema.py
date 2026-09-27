"""The single record format every dataset in this project is stored in."""
from __future__ import annotations

from dataclasses import asdict, dataclass

GOLD_LABELS = ("safe", "unsafe")


@dataclass(frozen=True)
class PromptRecord:
    """One English prompt.

    uid          stable id, e.g. "advbench-0007"; later steps (translation, model
                 responses, features) are joined back to the prompt by this id.
    source       which dataset it came from ("advbench", "xstest", ...).
    source_index position or id in the original file, so it can be traced back.
    text         the English prompt.
    gold_label   the dataset's own ground truth: "safe" or "unsafe".
                 This is NOT the model's decision; that comes later (z label).
    category     sub-type where the source has one (XSTest type, OR-Bench category).
    translate    whether step 2 should translate this prompt.
    """

    uid: str
    source: str
    source_index: int
    text: str
    gold_label: str
    category: str | None = None
    translate: bool = False

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError(f"{self.uid}: empty prompt text")
        if self.gold_label not in GOLD_LABELS:
            raise ValueError(f"{self.uid}: gold_label must be one of {GOLD_LABELS}, got {self.gold_label!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromptRecord":
        return cls(**d)
