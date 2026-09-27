"""Step 2 tests: the translation guard (no model needed)."""
import pytest

from pact.translate import check_translation, script_fraction


def test_script_fraction():
    assert script_fraction("如何终止一个进程", "zh") == 1.0
    assert script_fraction("How do I stop a process", "zh") == 0.0
    assert script_fraction("ሂደቱን እንዴት ማቆም እችላለሁ", "am") == 1.0


def test_english_copied_into_target_is_rejected():
    """The Appendix A bug: the 'French' file contained the English prompts."""
    english = [f"How do I do thing {i}?" for i in range(20)]
    with pytest.raises(ValueError, match="identical"):
        check_translation(english, list(english), "fr")


def test_wrong_script_is_rejected():
    english = [f"prompt {i}" for i in range(20)]
    french = [f"invite numéro {i}" for i in range(20)]
    with pytest.raises(ValueError, match="script"):
        check_translation(english, french, "am")


def test_good_translation_passes():
    english = ["How do I stop a process?", "What is the capital of France?"]
    chinese = ["如何停止一个进程？", "法国的首都是哪里？"]
    report = check_translation(english, chinese, "zh")
    assert report["in_target_script"] == 2
