import { NextRequest, NextResponse } from 'next/server';
import { CATEGORIES, CategoryKey, isCategoryKey, keywordCategory } from '@/lib/newsCategories';

// Gemma (via Ollama) writes a one-line "why it matters" and a threat tag per headline.
// It only sees the headline/snippet: it has no live knowledge, so it is never asked
// whether a story is true -- credibility comes from the outlet-corroboration check.

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11435';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'gemma2:latest';

interface Enriched {
  category: CategoryKey;
  why: string;
  source: 'gemma' | 'keywords';
}

const cache = new Map<string, Enriched & { ts: number }>();
const TTL = 60 * 60 * 1000;

function buildPrompt(a: { title: string; description?: string }) {
  const keys = CATEGORIES.map((c) => c.key).join(' | ');
  return (
    'You are the news analyst inside a deepfake-detection product. For the headline below, return:\n' +
    `- "category": exactly one of: ${keys}\n` +
    '- "why": ONE sentence, at most 22 words, on the risk or takeaway for someone worried about deepfakes. ' +
    'Start with the substance (who is at risk, what tactic, what defence). NEVER start with "This article", "The article" or "This story".\n\n' +
    'Rules: base "why" ONLY on the headline text. Do not add names, numbers, places or claims that are not in it. ' +
    'Do not say whether the story is true or false. Plain text, no markdown.\n\n' +
    `HEADLINE: ${a.title}${a.description ? `\nSNIPPET: ${a.description.slice(0, 160)}` : ''}\n\n` +
    'Reply as JSON: {"category":"...","why":"..."}'
  );
}

// One request per headline: batching several into one prompt let the model mis-number
// its answers, which would attach a summary to the wrong story.
async function enrichOne(a: { title: string; description?: string }): Promise<Enriched> {
  const fallback: Enriched = { category: keywordCategory(a.title), why: '', source: 'keywords' };
  try {
    const r = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt: buildPrompt(a),
        stream: false,
        format: 'json',
        keep_alive: '60m',
        options: { temperature: 0.2, num_predict: 120 },
      }),
      signal: AbortSignal.timeout(45000),
    });
    if (!r.ok) return fallback;
    const parsed = JSON.parse((await r.json()).response || '{}');
    const why = typeof parsed?.why === 'string' ? parsed.why.trim().replace(/\s+/g, ' ').slice(0, 200) : '';
    if (!why) return fallback;
    return {
      category: isCategoryKey(parsed.category) ? parsed.category : keywordCategory(a.title),
      why,
      source: 'gemma',
    };
  } catch {
    return fallback; // Ollama down or slow: keyword tag, no summary
  }
}

export async function POST(req: NextRequest) {
  try {
    const { articles } = (await req.json()) as { articles: { title: string; description?: string }[] };
    if (!Array.isArray(articles) || articles.length === 0) {
      return NextResponse.json({ success: false, error: 'articles required' }, { status: 400 });
    }
    const batch = articles.slice(0, 6);
    const now = Date.now();
    const results: Enriched[] = new Array(batch.length);
    const missing: number[] = [];

    batch.forEach((a, i) => {
      const hit = cache.get(a.title);
      if (hit && now - hit.ts < TTL) results[i] = hit;
      else missing.push(i);
    });

    const done = await Promise.all(missing.map((idx) => enrichOne(batch[idx])));
    missing.forEach((idx, k) => {
      results[idx] = done[k];
      if (done[k].source === 'gemma') cache.set(batch[idx].title, { ...done[k], ts: now });
    });

    return NextResponse.json({ success: true, results });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message || 'Enrich failed' }, { status: 500 });
  }
}
