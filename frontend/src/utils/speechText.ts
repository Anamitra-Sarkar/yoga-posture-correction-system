/**
 * Pure helpers for spoken guidance in English, Hindi and Bengali.
 *
 * Why this exists: the original splitter only knew `. ! ?`. Hindi and Bengali end sentences with the danda
 * (U+0964 "।", U+0965 "॥"), so a long Hindi/Bengali message never got split and Chrome silently stopped
 * speaking it after ~15 s. Voice matching also ignored that Android reports locales as "hi_IN" (underscore).
 */
export type SpeechLang = "en" | "hi" | "bn";

// Locales we accept per UI language, best first.
const LOCALES: Record<SpeechLang, string[]> = {
  en: ["en-us", "en-gb", "en-in"],
  hi: ["hi-in"],
  bn: ["bn-in", "bn-bd"],
};

export const normLocale = (l: string): string => (l || "").replace(/_/g, "-").toLowerCase();

/** The BCP-47 tag to put on an utterance for this UI language. */
export const utteranceLang = (lang: SpeechLang): string =>
  ({ en: "en-US", hi: "hi-IN", bn: "bn-IN" } as const)[lang];

/** Best installed voice for the language: exact locale first, then any voice of the same language. */
export function pickVoice<T extends { lang: string }>(voices: T[], lang: SpeechLang): T | undefined {
  for (const loc of LOCALES[lang]) {
    const exact = voices.find((v) => normLocale(v.lang) === loc);
    if (exact) return exact;
  }
  return voices.find((v) => normLocale(v.lang).split("-")[0] === lang);
}

/** True when the voice list has loaded (non-empty) and nothing in it speaks this language. */
export const voiceMissing = (voices: { lang: string }[], lang: SpeechLang): boolean =>
  lang !== "en" && voices.length > 0 && !pickVoice(voices, lang);

/**
 * Split text into chunks of at most `maxChunk` characters for chained utterances: first at sentence ends
 * (. ! ? । ॥), and any single over-long sentence at word boundaries. Never cuts a word.
 */
export function splitForSpeech(text: string, maxChunk = 180): string[] {
  const t = (text || "").trim();
  if (!t) return [];
  if (t.length <= maxChunk) return [t];
  const sentences = t.match(/[^.!?।॥]+[.!?।॥]*/g) ?? [t];
  const out: string[] = [];
  for (const raw of sentences) {
    const s = raw.trim();
    if (!s) continue;
    if (s.length <= maxChunk) { out.push(s); continue; }
    let cur = "";
    for (const w of s.split(/\s+/)) {
      if (cur && (cur + " " + w).length > maxChunk) { out.push(cur); cur = w; }
      else cur = cur ? cur + " " + w : w;
    }
    if (cur) out.push(cur);
  }
  return out;
}
