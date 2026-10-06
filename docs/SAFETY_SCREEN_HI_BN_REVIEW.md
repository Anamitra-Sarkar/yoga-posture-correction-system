# Hindi / Bengali safety screen -- DRAFT for native-speaker review (2026-10-07)

**What it does.** When the optional language-model paraphrase is on, the model's rewritten cue is checked before anyone hears it. If it contains a word below, the paraphrase is thrown away and the **reviewed template** is used instead. Over-blocking is harmless (the template is the safe fallback); under-blocking is the risk. The English list (push, force, hurt, pain, stretch more, further) is applied to every language, plus the language's own list below. Code: `backend/app/services/correction.py` (`FORBIDDEN_HI`, `FORBIDDEN_BN`); tests: `backend/tests/test_safety_screen.py`.

**Status: NOT SHIPPED.** Branch `fix/safety-screen-hi-bn`. Please strike, add or change words; matching is a substring match after normalisation (Devanagari nukta ignored, so ज़ = ज, ड़ = ड).

## Hindi (`FORBIDDEN_HI`)
| Word | Meaning | Replaces English |
|---|---|---|
| दर्द | pain | pain |
| पीड़ा | pain, suffering | pain |
| चोट | injury, hurt | hurt |
| ज़ोर | force, effort ("ज़ोर लगाएँ", "ज़ोर से") | force / push |
| ज़बरदस्ती | forcibly | force |
| जबरन | forcibly | force |
| धक्का | shove, push | push |
| धकेल | push (धकेलें) | push |
| और खींच | pull / stretch more (और खींचें) | stretch more |
| ज़्यादा खींच | pull / stretch more (ज़्यादा खींचें) | stretch more |
| और तान | stretch more (और तानें) | stretch more |
| और आगे | further ahead | further |
| और दूर | further away | further |

## Bengali (`FORBIDDEN_BN`)
| Word | Meaning | Replaces English |
|---|---|---|
| ব্যথা | pain | pain |
| যন্ত্রণা | agony, pain | pain |
| আঘাত | injury, blow | hurt |
| চোট | injury, hurt | hurt |
| জোর | force (জোর করে, জোরে) | force / push |
| জবরদস্তি | coercion, forcing | force |
| ঠেল | push (ঠেলুন, ঠেলে) | push |
| আরও টান / আরো টান | pull / stretch more | stretch more |
| আরও প্রসারিত / আরো প্রসারিত | more stretched | stretch more |
| আরও সামনে / আরো সামনে | further forward | further |
| আরও এগিয়ে / আরো এগিয়ে | further ahead | further |
| আরও দূরে / আরো দূরে | further away | further |

## Deliberately NOT blocked (they appear in the reviewed templates, and the English list has no equivalent)
* Hindi "और" alone, "थोड़ा और ... मोड़ें" (bend a little more), "दबाएँ" (press).
* Bengali "আরও বাঁকুন" (bend a bit more), "চাপ দিন" (press -- the reviewed Downward Dog cue "হাতের উপর দৃঢ়ভাবে চাপ দিন" uses it; a first draft of this list blocked "চাপ" and a test caught it), "প্রসারিত করুন" alone (extend).
* Bare "pressure" words in both languages (दबाव / চাপ): English has no "pressure"/"press" entry either.

## Guard-rail
`test_every_reviewed_sentence_passes_the_screen` runs every template, default cue and escalation sentence in all three languages through the screen, so adding a word that a reviewed sentence uses fails the tests instead of silently disabling that cue's paraphrase.

## Known limit that remains
A word list cannot catch every unsafe sentence in an open-ended paraphrase. The strongest safeguards stay the reviewed templates and the escalation back-off ("ease out of the shape..."). Turning the paraphrase off for Hindi/Bengali (templates only) is the zero-risk alternative.
