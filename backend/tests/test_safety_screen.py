"""Stage-3 safety screen (correction.py): English list unchanged, Hindi and Bengali lists added (DRAFT, pending native-speaker review).
Also proves that EVERY reviewed template, default cue and escalation sentence passes the screen -- otherwise a paraphrase that merely
repeated the template would be rejected and the fallback would be doing needless work."""
import pytest

from app.services import correction as c


def test_english_behaviour_is_unchanged():
    for bad in ("Push deeper into the pose.", "Do not force it, it may hurt.", "You will feel pain.", "Stretch more.", "Reach further."):
        assert c.violates_safety_screen(bad, "en")
    for ok in ("Bend your left knee more to bring it directly over your ankle.", "Pose alignment correct. Keep breathing steadily."):
        assert not c.violates_safety_screen(ok, "en")


@pytest.mark.parametrize("word", c.FORBIDDEN_HI)
def test_every_hindi_word_is_caught_inside_a_sentence(word):
    assert c.violates_safety_screen(f"अपने घुटने को सीधा रखें, {word} न करें।", "hi")


@pytest.mark.parametrize("word", c.FORBIDDEN_BN)
def test_every_bengali_word_is_caught_inside_a_sentence(word):
    assert c.violates_safety_screen(f"আপনার হাঁটু সোজা রাখুন, {word} করবেন না।", "bn")


def test_hindi_nukta_spellings_are_treated_alike():
    assert c.violates_safety_screen("ज़ोर लगाएँ", "hi") and c.violates_safety_screen("जोर लगाएँ", "hi")


def test_english_word_inside_a_hindi_or_bengali_answer_is_still_caught():
    assert c.violates_safety_screen("घुटने को push करें", "hi") and c.violates_safety_screen("হাঁটু force করুন", "bn")


def _all_reviewed_sentences():
    for pose, joints in c.BIOMECHANICAL_TEMPLATES.items():
        for joint, kinds in joints.items():
            for kind, langs in kinds.items():
                for lang, text in langs.items():
                    yield f"template:{pose}/{joint}/{kind}/{lang}", lang, text
    for lang, text in c.DEFAULT_CORRECTIONS.items():
        yield f"default:{lang}", lang, text
    for tier, langs in c._ESCALATION.items():
        for lang, text in langs.items():
            yield f"escalation:{tier}/{lang}", lang, text.format(base="", deg=24)


def test_every_reviewed_sentence_passes_the_screen():
    bad = [(k, t) for k, lang, t in _all_reviewed_sentences() if c.violates_safety_screen(t, lang)]
    assert not bad, bad
