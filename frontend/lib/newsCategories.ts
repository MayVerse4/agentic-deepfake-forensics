// Threat categories for the trending feed. Shared by the enrich API route (validates
// Gemma's tag, falls back to keywords) and the TrendingPage (colours / labels).

export type CategoryKey = 'fraud' | 'voice' | 'face' | 'politics' | 'policy' | 'detection' | 'other';

export const CATEGORIES: { key: CategoryKey; label: string; color: string }[] = [
  { key: 'fraud', label: 'Fraud & Scams', color: '#F59E0B' },
  { key: 'voice', label: 'Voice Cloning', color: '#A855F7' },
  { key: 'face', label: 'Face Swap & Video', color: '#3B82F6' },
  { key: 'politics', label: 'Politics & Elections', color: '#EF4444' },
  { key: 'policy', label: 'Law & Policy', color: '#14B8A6' },
  { key: 'detection', label: 'Detection & Research', color: '#10B981' },
  { key: 'other', label: 'General', color: '#94A3B8' },
];

export const CATEGORY_BY_KEY = Object.fromEntries(CATEGORIES.map((c) => [c.key, c])) as Record<CategoryKey, (typeof CATEGORIES)[number]>;

export function isCategoryKey(x: unknown): x is CategoryKey {
  return typeof x === 'string' && x in CATEGORY_BY_KEY;
}

const RULES: [CategoryKey, RegExp][] = [
  ['voice', /voice.?clon|clone[sd]? (his|her|their|a) voice|audio deepfake|voice deepfake|cloned voice|voice scam/i],
  ['fraud', /scam|fraud|phish|extort|blackmail|impersonat|heist|swindl|money|bank|crypto|romance|sextortion/i],
  ['politics', /election|politic|campaign|president|minister|senator|candidate|vote|parliament|government/i],
  ['policy', /law|bill\b|legislat|regulat|ban\b|court|lawsuit|sued|act\b|policy|fine[sd]?\b|penalt|congress|eu\b/i],
  ['detection', /detect|research|study|watermark|authentic|verif|forensic|tool|model|benchmark|classifier|c2pa/i],
  ['face', /face.?swap|deepfake video|video|celebrity|nude|porn|image|photo|avatar|likeness/i],
];

export function keywordCategory(title: string): CategoryKey {
  for (const [key, re] of RULES) if (re.test(title)) return key;
  return 'other';
}
