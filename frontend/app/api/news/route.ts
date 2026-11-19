import { NextResponse } from 'next/server';

interface NewsItem {
  title: string;
  description: string;
  url: string;
  image: string | null;
  source: string;
  publishedAt: string;
}

// Deepfake/AI-fraud-focused queries -- this is a deepfake detection product,
// so the feed should be about synthetic media, not generic world news.
const TOPIC_QUERIES = [
  'deepfake OR "AI-generated video" OR "synthetic media" fraud',
  '"voice clone" OR "face swap" scam',
];

async function scrapeGoogleNews(): Promise<NewsItem[]> {
  const allItems: NewsItem[] = [];

  for (const query of TOPIC_QUERIES) {
  try {
    const res = await fetch(
      `https://news.google.com/rss/search?q=${encodeURIComponent(query)}&hl=en-US&gl=US&ceid=US:en`,
      {
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
        next: { revalidate: 120 },
      }
    );

    if (res.ok) {
      const xml = await res.text();
      const itemRegex = /<item>([\s\S]*?)<\/item>/g;
      let match;

      while ((match = itemRegex.exec(xml)) !== null && allItems.length < 20) {
        const itemXml = match[1];

        const titleMatch = itemXml.match(/<title><!\[CDATA\[([\s\S]*?)\]\]><\/title>/) ||
          itemXml.match(/<title>([\s\S]*?)<\/title>/);
        const linkMatch = itemXml.match(/<link>([\s\S]*?)<\/link>/);
        const descMatch = itemXml.match(/<description><!\[CDATA\[([\s\S]*?)\]\]><\/description>/) ||
          itemXml.match(/<description>([\s\S]*?)<\/description>/);
        const pubDateMatch = itemXml.match(/<pubDate>([\s\S]*?)<\/pubDate>/);
        const sourceMatch = itemXml.match(/<source[^>]*>([\s\S]*?)<\/source>/);
        const sourceUrlMatch = itemXml.match(/<source[^>]*\surl="([^"]*)"/);

        const titleRaw = titleMatch?.[1] || '';
        const descRaw = descMatch?.[1] || '';

        const decode = (str: string) => str.replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&#x27;/g, "'");

        const title = decode(titleRaw).replace(/<[^>]*>/g, '').trim();
        const url = (linkMatch?.[1] || '').trim();
        let rawDesc = decode(descRaw).replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
        const publishedAt = pubDateMatch?.[1] || '';
        const source = (sourceMatch?.[1] || '').replace(/<[^>]*>/g, '').trim();

        if (!title || !url) continue;

        // Google News RSS descriptions are just "<title>  <source>" or a
        // clustered list of related headlines -- never a real summary, so a
        // description that starts with the title (or the whole thing, for
        // clusters) adds nothing and reads as duplicated/garbled text.
        if (rawDesc.toLowerCase().startsWith(title.toLowerCase().slice(0, 30))) {
          rawDesc = '';
        }

        // `url` is Google's redirect link (always news.google.com), not the
        // publisher's own domain -- the real domain lives in <source url="">.
        const hostname = (() => {
          try {
            return new URL(sourceUrlMatch?.[1] || url).hostname.replace('www.', '');
          } catch {
            return source || 'web';
          }
        })();

        // The feed never carries a real photo for search results; use the
        // outlet's real favicon (Google's public favicon service) rather
        // than showing a blank placeholder for every single card.
        const imgMatch = decode(descRaw).match(/<img[^>]*src="([^"]+)"/);
        const image = imgMatch?.[1] || `https://www.google.com/s2/favicons?sz=128&domain=${hostname}`;

        allItems.push({
          title,
          description: rawDesc.slice(0, 200),
          url,
          image,
          source: source || hostname,
          publishedAt,
        });
      }
    }
  } catch {
    // this query failed, continue with the others
  }
  }

  if (allItems.length < 5) {
    try {
      const ddgItems = await fetchDuckDuckGoNews();
      allItems.push(...ddgItems);
    } catch {
      // DuckDuckGo also failed
    }
  }

  // Deduplicate by title
  const seen = new Set<string>();
  const unique = allItems.filter((item) => {
    const key = item.title.toLowerCase().slice(0, 60);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return unique.slice(0, 20);
}

async function fetchDuckDuckGoNews(): Promise<NewsItem[]> {
  const queries = TOPIC_QUERIES;
  const allItems: NewsItem[] = [];

  for (const query of queries) {
    try {
      const encodedQuery = encodeURIComponent(query);
      const res = await fetch(
        `https://html.duckduckgo.com/html/?q=${encodedQuery}`,
        {
          headers: {
            'User-Agent':
              'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          },
          next: { revalidate: 300 },
        }
      );

      if (!res.ok) continue;
      const html = await res.text();

      const resultRegex =
        /<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g;
      const snippetRegex =
        /<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
      const iconRegex =
        /<img[^>]*class="result__icon__img"[^>]*src="([^"]*)"/g;

      let m;
      const titles: { title: string; url: string }[] = [];
      const snippets: string[] = [];
      const icons: string[] = [];

      while ((m = resultRegex.exec(html)) !== null) {
        const rawUrl = m[1] || '';
        const title = (m[2] || '')
          .replace(/<[^>]*>/g, '')
          .replace(/&amp;/g, '&')
          .replace(/&lt;/g, '<')
          .replace(/&gt;/g, '>')
          .replace(/&quot;/g, '"')
          .replace(/&#x27;/g, "'")
          .trim();

        let url = rawUrl;
        const uddgMatch = rawUrl.match(/uddg=([^&]*)/);
        if (uddgMatch) url = decodeURIComponent(uddgMatch[1]);

        if (title && url) titles.push({ title, url });
      }

      while ((m = snippetRegex.exec(html)) !== null) {
        const snippet = (m[1] || '')
          .replace(/<[^>]*>/g, '')
          .replace(/&amp;/g, '&')
          .replace(/&lt;/g, '<')
          .replace(/&gt;/g, '>')
          .replace(/&quot;/g, '"')
          .replace(/&#x27;/g, "'")
          .trim();
        snippets.push(snippet);
      }

      while ((m = iconRegex.exec(html)) !== null) {
        icons.push(m[1] || '');
      }

      for (let i = 0; i < Math.min(titles.length, 10); i++) {
        const hostname = (() => {
          try {
            return new URL(titles[i].url).hostname.replace('www.', '');
          } catch {
            return 'web';
          }
        })();

        const iconUrl = icons[i] || null;
        const image = iconUrl && iconUrl.startsWith('//') ? `https:${iconUrl}` : iconUrl;

        allItems.push({
          title: titles[i].title,
          description: snippets[i] || '',
          url: titles[i].url,
          image: image && image.includes('icon') ? null : image,
          source: hostname,
          publishedAt: new Date().toISOString(),
        });
      }
    } catch {
      continue;
    }
  }

  return allItems;
}

export async function GET() {
  try {
    const news = await scrapeGoogleNews();
    return NextResponse.json({ success: true, data: news, fetchedAt: new Date().toISOString() });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to fetch news' },
      { status: 500 }
    );
  }
}
