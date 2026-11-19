'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Newspaper,
  ExternalLink,
  Globe,
  RefreshCw,
  AlertTriangle,
  Clock,
  Circle,
  ChevronDown,
  ChevronUp,
  Shield,
  CheckCircle,
  HelpCircle,
  XCircle,
  Search,
  Link2,
  Sparkles,
  ScanFace,
  Mic,
  Landmark,
  Scale,
  Microscope,
  Banknote,
  Flame,
  ArrowRight,
} from 'lucide-react';
import { CATEGORIES, CATEGORY_BY_KEY, CategoryKey, keywordCategory } from '@/lib/newsCategories';

interface NewsArticle {
  title: string;
  description: string;
  url: string;
  image: string | null;
  source: string;
  publishedAt: string;
}

interface VerificationResult {
  title: string;
  corroborating_sources: string[];
  source_count: number;
  verification_status: 'Verified' | 'Likely Real' | 'Partially Verified' | 'Unverified' | 'Single Source';
  confidence_score: number;
  label: string;
}

interface Enriched {
  category: CategoryKey;
  why: string;
  source: 'gemma' | 'keywords';
}

interface TrendingPageProps {
  isFullPage?: boolean;
  onNavigate?: (page: string) => void;
}

const CATEGORY_ICON: Record<CategoryKey, React.ComponentType<{ className?: string; style?: React.CSSProperties }>> = {
  fraud: Banknote,
  voice: Mic,
  face: ScanFace,
  politics: Landmark,
  policy: Scale,
  detection: Microscope,
  other: Newspaper,
};

function timeAgo(dateStr: string): string {
  if (!dateStr) return '';
  try {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ago`;
  } catch {
    return '';
  }
}

// The verify route counts how many independent outlets carry a story. That measures
// coverage, not truth, so the UI says "widely reported", never "verified real".
const COVERAGE_LABEL: Record<string, string> = {
  Verified: 'Widely reported',
  'Likely Real': 'Corroborated',
  'Partially Verified': 'Limited coverage',
  Unverified: 'Little coverage',
  'Single Source': 'Single source',
};

function statusColor(status: string): string {
  switch (status) {
    case 'Verified': return 'text-green-400';
    case 'Likely Real': return 'text-emerald-400';
    case 'Partially Verified': return 'text-yellow-400';
    case 'Unverified': return 'text-orange-400';
    case 'Single Source': return 'text-blue-400';
    default: return 'text-white/60';
  }
}

function statusBadgeStyle(status: string): string {
  switch (status) {
    case 'Verified': return 'bg-green-500/15 text-green-400 shadow-[0_0_10px_rgba(34,197,94,0.2)]';
    case 'Likely Real': return 'bg-emerald-500/15 text-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.15)]';
    case 'Partially Verified': return 'bg-yellow-500/15 text-yellow-400 shadow-[0_0_10px_rgba(234,179,8,0.15)]';
    case 'Unverified': return 'bg-orange-500/15 text-orange-400 shadow-[0_0_10px_rgba(249,115,22,0.15)]';
    case 'Single Source': return 'bg-blue-500/15 text-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.15)]';
    default: return 'dark:bg-white/10 bg-black/10 text-white/60';
  }
}

function statusBorderColor(status: string): string {
  switch (status) {
    case 'Verified': return 'border-green-500/20';
    case 'Likely Real': return 'border-emerald-500/20';
    case 'Partially Verified': return 'border-yellow-500/20';
    case 'Unverified': return 'border-orange-500/20';
    case 'Single Source': return 'border-blue-500/20';
    default: return 'dark:border-white/[0.08] border-black/[0.08]';
  }
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'Verified': return <CheckCircle className="w-3 h-3 text-green-400" />;
    case 'Likely Real': return <CheckCircle className="w-3 h-3 text-emerald-400" />;
    case 'Partially Verified': return <HelpCircle className="w-3 h-3 text-yellow-400" />;
    case 'Unverified': return <AlertTriangle className="w-3 h-3 text-orange-400" />;
    case 'Single Source': return <XCircle className="w-3 h-3 text-blue-400" />;
    default: return <HelpCircle className="w-3 h-3 text-white/60" />;
  }
}

const glass = 'dark:bg-white/[0.03] bg-black/[0.03] backdrop-blur-xl border dark:border-white/[0.08] border-black/[0.08]';

export default function TrendingPage({ isFullPage = true, onNavigate }: TrendingPageProps) {
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
  const [verifications, setVerifications] = useState<Map<string, VerificationResult>>(new Map());
  const [verifying, setVerifying] = useState(false);
  const [enrichments, setEnrichments] = useState<Map<string, Enriched>>(new Map());
  const [enriching, setEnriching] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<CategoryKey | 'all'>('all');
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const enrichedRef = useRef<Set<string>>(new Set());

  const verifyArticles = useCallback(async (articleList: NewsArticle[]) => {
    if (articleList.length === 0) return;
    setVerifying(true);
    try {
      // Verify in batches of 10
      for (let i = 0; i < articleList.length; i += 10) {
        const batch = articleList.slice(i, i + 10);
        const res = await fetch('/api/news/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            articles: batch.map(a => ({ title: a.title, source: a.source }))
          }),
        });
        const data = await res.json();
        if (data?.success && Array.isArray(data.results)) {
          setVerifications(prev => {
            const next = new Map(prev);
            data.results.forEach((r: VerificationResult, idx: number) => {
              const article = batch[idx];
              if (article) {
                next.set(article.title, r);
              }
            });
            return next;
          });
        }
      }
    } catch {
      // Verification failed silently, articles still show without verification
    }
    setVerifying(false);
  }, []);

  // Gemma writes the one-line summary + threat tag, a few headlines at a time so
  // cards fill in progressively instead of after one long wait.
  const enrichArticles = useCallback(async (articleList: NewsArticle[]) => {
    const todo = articleList.filter((a) => !enrichedRef.current.has(a.title));
    if (todo.length === 0) return;
    setEnriching(true);
    try {
      for (let i = 0; i < todo.length; i += 5) {
        const batch = todo.slice(i, i + 5);
        const res = await fetch('/api/news/enrich', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ articles: batch.map((a) => ({ title: a.title, description: a.description })) }),
        });
        const data = await res.json();
        if (data?.success && Array.isArray(data.results)) {
          setEnrichments((prev) => {
            const next = new Map(prev);
            data.results.forEach((r: Enriched, idx: number) => {
              if (batch[idx]) next.set(batch[idx].title, r);
            });
            return next;
          });
          batch.forEach((a) => enrichedRef.current.add(a.title));
        }
      }
    } catch {
      // cards keep their keyword tag and simply have no summary line
    }
    setEnriching(false);
  }, []);

  const fetchNews = useCallback(async (showLoadingState = true) => {
    if (showLoadingState) setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/news');
      const data = await res.json();
      if (data?.success && Array.isArray(data.data)) {
        setArticles(data.data);
        setLastUpdated(new Date());
        setFailedImages(new Set());
        // Background enrichment
        verifyArticles(data.data);
        enrichArticles(data.data);
      } else {
        setError(data?.error || 'Failed to fetch news');
      }
    } catch {
      setError('Failed to fetch news. Please check your connection.');
    }
    setLoading(false);
  }, [verifyArticles, enrichArticles]);

  useEffect(() => {
    fetchNews();
    intervalRef.current = setInterval(() => fetchNews(false), 90000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchNews]);

  const categoryOf = useCallback(
    (a: NewsArticle): CategoryKey => enrichments.get(a.title)?.category ?? keywordCategory(a.title),
    [enrichments]
  );

  // ---- stats ----
  const stats = useMemo(() => {
    const counts = new Map<CategoryKey, number>();
    articles.forEach((a) => counts.set(categoryOf(a), (counts.get(categoryOf(a)) || 0) + 1));
    const outlets = new Set(articles.map((a) => a.source)).size;
    const wide = articles.filter((a) => (verifications.get(a.title)?.source_count ?? 0) >= 3).length;
    const top = Array.from(counts.entries()).filter(([k]) => k !== 'other').sort((a, b) => b[1] - a[1])[0];
    return { counts, outlets, wide, top };
  }, [articles, verifications, categoryOf]);

  const filtered = useMemo(
    () => (filter === 'all' ? articles : articles.filter((a) => categoryOf(a) === filter)),
    [articles, filter, categoryOf]
  );

  const hero = filtered[0];
  const rest = filtered.slice(1);
  const displayArticles = rest.slice(0, showAll ? 19 : 9);

  const handleImageError = (key: string) => {
    setFailedImages((prev) => new Set(prev).add(key));
  };

  const toggleSources = (key: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setExpandedSources(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const goAnalyze = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onNavigate?.('analysis');
  };

  // ---- shared card pieces ----
  const CategoryBadge = ({ article }: { article: NewsArticle }) => {
    const cat = CATEGORY_BY_KEY[categoryOf(article)];
    const Icon = CATEGORY_ICON[cat.key];
    return (
      <span
        className="inline-flex items-center gap-1 text-[10px] font-sans font-semibold px-2 py-0.5 rounded-full border"
        style={{ color: cat.color, background: `${cat.color}1a`, borderColor: `${cat.color}40` }}
      >
        <Icon className="w-2.5 h-2.5" />
        {cat.label}
      </span>
    );
  };

  const WhyLine = ({ article, large = false }: { article: NewsArticle; large?: boolean }) => {
    const e = enrichments.get(article.title);
    if (e?.why) {
      return (
        <p className={`flex items-start gap-1.5 font-sans leading-relaxed text-foreground/80 ${large ? 'text-sm mb-3' : 'text-xs mb-2'}`}>
          <Sparkles className="w-3 h-3 mt-0.5 shrink-0 text-[#A855F7]" />
          <span><span className="text-[#A855F7] font-semibold">Why it matters · </span>{e.why}</span>
        </p>
      );
    }
    if (enriching && !e) {
      return (
        <div className={`flex items-center gap-1.5 ${large ? 'mb-3' : 'mb-2'}`}>
          <Sparkles className="w-3 h-3 text-[#A855F7]/50 animate-pulse" />
          <div className="h-2.5 w-2/3 rounded-full dark:bg-white/[0.06] bg-black/[0.06] animate-pulse" />
        </div>
      );
    }
    return null;
  };

  const CoverageRow = ({ article, k }: { article: NewsArticle; k: string }) => {
    const verification = verifications.get(article.title);
    const isVerifying = !verification && verifying;
    const showSources = expandedSources.has(k);
    if (isVerifying) {
      return (
        <div className="flex items-center gap-1.5">
          <Shield className="w-3 h-3 text-muted-foreground/40" />
          <RefreshCw className="w-2.5 h-2.5 text-muted-foreground/40 animate-spin" />
          <span className="text-[10px] font-sans text-muted-foreground/40">Checking coverage...</span>
        </div>
      );
    }
    if (!verification) {
      return (
        <div className="flex items-center gap-1.5">
          <Shield className="w-3 h-3 text-muted-foreground/40" />
          <span className="text-[10px] font-sans text-muted-foreground/40">Coverage check pending</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 flex-wrap">
        <Shield className="w-3 h-3 text-muted-foreground/40" />
        <StatusIcon status={verification.verification_status} />
        <span className={`text-[10px] font-sans font-bold px-2 py-0.5 rounded-full ${statusBadgeStyle(verification.verification_status)}`}>
          {COVERAGE_LABEL[verification.verification_status] || verification.verification_status}
        </span>
        {verification.source_count > 0 && (
          <button
            onClick={(e) => toggleSources(k, e)}
            className={`text-[10px] font-sans flex items-center gap-1 hover:text-foreground transition-colors ${statusColor(verification.verification_status)}`}
          >
            <Link2 className="w-2.5 h-2.5" />
            Reported by {verification.source_count} {verification.source_count === 1 ? 'outlet' : 'outlets'}
            {showSources ? <ChevronUp className="w-2.5 h-2.5" /> : <ChevronDown className="w-2.5 h-2.5" />}
          </button>
        )}
      </div>
    );
  };

  const SourcesPanel = ({ article, k }: { article: NewsArticle; k: string }) => {
    const verification = verifications.get(article.title);
    if (!expandedSources.has(k) || !verification || verification.corroborating_sources.length === 0) return null;
    return (
      <div className="mt-1 ml-4 mr-4 p-3 dark:bg-white/[0.02] bg-black/[0.02] border dark:border-white/[0.06] border-black/[0.06] rounded-xl">
        <div className="flex items-center gap-1.5 mb-2">
          <Link2 className="w-3 h-3 text-muted-foreground/60" />
          <span className="text-[11px] font-sans font-medium text-muted-foreground/80">
            Outlets carrying this story (coverage, not proof it is true):
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {verification.corroborating_sources.map((source, idx) => (
            <span
              key={idx}
              className="text-[10px] font-sans px-2 py-0.5 rounded-full dark:bg-white/[0.06] bg-black/[0.06] text-muted-foreground border dark:border-white/[0.08] border-black/[0.08]"
            >
              {source}
            </span>
          ))}
        </div>
      </div>
    );
  };

  const AnalyzeButton = () =>
    onNavigate ? (
      <button
        onClick={goAnalyze}
        className="text-[11px] font-sans font-medium flex items-center gap-1 text-[#3B82F6] hover:text-[#60A5FA] transition-colors"
      >
        <ScanFace className="w-3 h-3" />
        Analyze related media
        <ArrowRight className="w-3 h-3" />
      </button>
    ) : null;

  const Visual = ({ article, size }: { article: NewsArticle; size: 'hero' | 'card' }) => {
    const cat = CATEGORY_BY_KEY[categoryOf(article)];
    const Icon = CATEGORY_ICON[cat.key];
    const key = article.url;
    const showFavicon = article.image && !failedImages.has(key);
    const box = size === 'hero' ? 'w-full md:w-56 min-h-[150px]' : 'w-20 min-h-[7.5rem]';
    return (
      <div
        className={`${box} flex-shrink-0 relative flex items-center justify-center overflow-hidden`}
        style={{ background: `linear-gradient(135deg, ${cat.color}33 0%, ${cat.color}0d 70%)` }}
      >
        <Icon className={size === 'hero' ? 'w-14 h-14' : 'w-7 h-7'} style={{ color: `${cat.color}cc` }} />
        <div className="absolute bottom-2 left-2 w-6 h-6 rounded-md bg-white/90 flex items-center justify-center shadow">
          {showFavicon ? (
            <img src={article.image!} alt="" className="w-4 h-4 object-contain" onError={() => handleImageError(key)} />
          ) : (
            <Globe className="w-3.5 h-3.5 text-black/40" />
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className={`flex items-center justify-between mb-5 ${glass} rounded-2xl px-6 py-4`}>
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#3B82F6] to-[#1D4ED8] flex items-center justify-center shadow-[0_0_15px_rgba(59,130,246,0.25)]">
              <Newspaper className="w-4 h-4 text-white" />
            </div>
            <h2 className="text-2xl font-bold font-sans tracking-tight text-foreground">
              Deepfake & AI-Fraud News
            </h2>
          </div>
          <p className="text-sm text-muted-foreground font-sans mt-1 ml-11">
            Live synthetic-media coverage, tagged and summarised by a local Gemma 2 model
          </p>
        </div>
        <div className="flex items-center gap-4">
          {(verifying || enriching) && (
            <div className="hidden sm:flex items-center gap-1.5 text-xs text-[#3B82F6]/70 font-sans">
              {enriching ? <Sparkles className="w-3 h-3 animate-pulse" /> : <Search className="w-3 h-3 animate-pulse" />}
              <span>{enriching ? 'Gemma is summarising...' : 'Checking coverage...'}</span>
            </div>
          )}
          {lastUpdated && (
            <div className="hidden md:flex items-center gap-1.5 text-xs text-muted-foreground/70 font-sans">
              <Clock className="w-3 h-3" />
              <span>Updated {lastUpdated.toLocaleTimeString()}</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 bg-green-500/10 px-2.5 py-1 rounded-full">
            <Circle className="w-2 h-2 text-green-400 animate-pulse" />
            <span className="text-xs text-green-400 font-sans font-medium">Live</span>
          </div>
        </div>
      </div>

      {/* How credibility is judged */}
      <div className={`${glass} rounded-2xl px-5 py-3 mb-4 flex flex-col gap-2`}>
        <div className="flex items-center gap-x-4 gap-y-1.5 flex-wrap">
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground/70 font-sans font-medium">
            <Shield className="w-3 h-3" /> Coverage check (outlets carrying the story):
          </span>
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-sans"><CheckCircle className="w-2.5 h-2.5 text-green-400" />Widely reported · 5+ outlets</span>
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-sans"><CheckCircle className="w-2.5 h-2.5 text-emerald-400" />Corroborated · 3+</span>
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-sans"><HelpCircle className="w-2.5 h-2.5 text-yellow-400" />Limited · 2</span>
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-sans"><AlertTriangle className="w-2.5 h-2.5 text-orange-400" />Little coverage · 0-1</span>
        </div>
        <p className="text-[11px] text-muted-foreground/60 font-sans leading-relaxed">
          Credibility comes from counting independent outlets, a rule-based check. It shows how widely a story is reported, not that it is true.
          The Gemma 2 model only tags each story and writes the one-line summary from its headline.
        </p>
      </div>

      {/* Stats strip */}
      {articles.length > 0 && (
        <div className={`${glass} rounded-2xl p-4 mb-4`}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Stories tracked', value: String(articles.length), sub: 'from live news feeds' },
              { label: 'Outlets', value: String(stats.outlets), sub: 'distinct publishers' },
              { label: 'Widely reported', value: verifications.size ? String(stats.wide) : '…', sub: 'carried by 3+ outlets' },
              {
                label: 'Top threat',
                value: stats.top ? CATEGORY_BY_KEY[stats.top[0]].label : '—',
                sub: stats.top ? `${stats.top[1]} stories` : '',
              },
            ].map((t) => (
              <div key={t.label} className="rounded-xl dark:bg-white/[0.03] bg-black/[0.03] px-3.5 py-2.5">
                <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground font-sans">{t.label}</p>
                <p className="text-lg font-bold font-sans text-foreground leading-tight mt-0.5 truncate">{t.value}</p>
                <p className="text-[10px] text-muted-foreground/60 font-sans">{t.sub}</p>
              </div>
            ))}
          </div>
          {/* category distribution bar */}
          <div className="mt-3 flex h-1.5 w-full rounded-full overflow-hidden dark:bg-white/[0.06] bg-black/[0.06]">
            {CATEGORIES.filter((c) => stats.counts.get(c.key)).map((c) => (
              <div
                key={c.key}
                className="h-full transition-all duration-500"
                style={{ width: `${((stats.counts.get(c.key) || 0) / articles.length) * 100}%`, background: c.color }}
                title={`${c.label}: ${stats.counts.get(c.key)}`}
              />
            ))}
          </div>
        </div>
      )}

      {/* Filter chips */}
      {articles.length > 0 && (
        <div className="flex items-center gap-2 mb-5 flex-wrap">
          {[{ key: 'all' as const, label: 'All', color: '#3B82F6', count: articles.length },
            ...CATEGORIES.filter((c) => stats.counts.get(c.key)).map((c) => ({ key: c.key, label: c.label, color: c.color, count: stats.counts.get(c.key) || 0 }))]
            .map((c) => {
              const active = filter === c.key;
              return (
                <button
                  key={c.key}
                  onClick={() => { setFilter(c.key); setShowAll(false); }}
                  className="text-xs font-sans font-medium px-3 py-1.5 rounded-full border transition-all duration-200"
                  style={active
                    ? { color: '#fff', background: c.color, borderColor: c.color, boxShadow: `0 0 12px ${c.color}55` }
                    : { color: c.color, background: `${c.color}12`, borderColor: `${c.color}33` }}
                >
                  {c.label} <span className="opacity-70 ml-0.5">{c.count}</span>
                </button>
              );
            })}
        </div>
      )}

      {/* Loading State */}
      {loading && articles.length === 0 && (
        <div className={`${glass} rounded-2xl p-12 text-center mb-6`}>
          <RefreshCw className="w-8 h-8 mx-auto mb-4 text-[#3B82F6] animate-spin" />
          <p className="text-sm font-sans text-foreground/70">Fetching latest news...</p>
          <p className="text-xs font-sans text-muted-foreground/50 mt-1">
            Scanning live sources, then tagging and checking coverage
          </p>
        </div>
      )}

      {/* Error State */}
      {error && articles.length === 0 && (
        <div className="border border-yellow-500/20 rounded-2xl bg-yellow-500/5 backdrop-blur-xl p-8 text-center mb-6">
          <AlertTriangle className="w-8 h-8 mx-auto mb-3 text-yellow-500" />
          <p className="text-sm font-sans text-foreground/70 mb-3">{error}</p>
          <button
            onClick={() => fetchNews()}
            className="text-xs font-sans text-yellow-400 hover:text-yellow-300 underline underline-offset-4 transition-colors"
          >
            Try again
          </button>
        </div>
      )}

      {/* Hero */}
      {hero && (() => {
        const v = verifications.get(hero.title);
        const cat = CATEGORY_BY_KEY[categoryOf(hero)];
        return (
          <div className="mb-4">
            <a href={hero.url} target="_blank" rel="noopener noreferrer" className="block group">
              <div
                className={`rounded-2xl overflow-hidden border dark:bg-white/[0.04] bg-black/[0.03] transition-all duration-300 hover:shadow-[0_0_28px_rgba(59,130,246,0.12)]`}
                style={{ borderColor: `${cat.color}40` }}
              >
                <div className="flex flex-col md:flex-row">
                  <Visual article={hero} size="hero" />
                  <div className="flex-1 p-5 min-w-0">
                    <div className="flex items-center gap-2 mb-2.5 flex-wrap">
                      <span className="inline-flex items-center gap-1 text-[10px] font-sans font-bold uppercase tracking-wider text-orange-400">
                        <Flame className="w-3 h-3" /> Top story
                      </span>
                      <CategoryBadge article={hero} />
                    </div>
                    <h3 className="text-lg font-bold font-sans text-foreground group-hover:text-[#3B82F6] transition-colors leading-snug mb-2.5 line-clamp-3">
                      {hero.title}
                    </h3>
                    <WhyLine article={hero} large />
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-[11px] font-sans text-muted-foreground dark:bg-white/[0.06] bg-black/[0.06] px-2.5 py-0.5 rounded-full">{hero.source}</span>
                      {hero.publishedAt && (
                        <span className="text-[11px] text-muted-foreground/60 font-sans flex items-center gap-1">
                          <Clock className="w-2.5 h-2.5" />{timeAgo(hero.publishedAt)}
                        </span>
                      )}
                      <span className="text-[11px] text-muted-foreground/50 font-sans flex items-center gap-1 ml-auto group-hover:text-[#3B82F6]/70 transition-colors">
                        <ExternalLink className="w-2.5 h-2.5" /> Read
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-3 flex-wrap">
                      <CoverageRow article={hero} k={hero.url} />
                      <AnalyzeButton />
                    </div>
                  </div>
                </div>
              </div>
            </a>
            <SourcesPanel article={hero} k={hero.url} />
          </div>
        );
      })()}

      {/* News Feed */}
      <div className="space-y-3">
        {displayArticles.map((article, i) => {
          const v = verifications.get(article.title);
          const k = article.url;
          return (
            <div key={`${article.url}-${i}`} className="block group">
              <a href={article.url} target="_blank" rel="noopener noreferrer" className="block">
                <div className={`border ${v ? statusBorderColor(v.verification_status) : 'dark:border-white/[0.08] border-black/[0.08]'} rounded-2xl dark:bg-white/[0.03] bg-black/[0.03] backdrop-blur-xl hover:dark:bg-white/[0.06] hover:dark:border-white/[0.15] hover:border-black/[0.15] hover:shadow-[0_0_20px_rgba(59,130,246,0.08)] transition-all duration-300 cursor-pointer overflow-hidden`}>
                  <div className="flex">
                    <Visual article={article} size="card" />
                    <div className="flex-1 p-4 min-w-0">
                      <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                        <CategoryBadge article={article} />
                        <span className="text-[11px] font-sans text-muted-foreground dark:bg-white/[0.06] bg-black/[0.06] px-2.5 py-0.5 rounded-full">
                          {article.source}
                        </span>
                        {article.publishedAt && (
                          <span className="text-[11px] text-muted-foreground/50 font-sans flex items-center gap-1">
                            <Clock className="w-2.5 h-2.5" />
                            {timeAgo(article.publishedAt)}
                          </span>
                        )}
                        <span className="text-[11px] text-muted-foreground/40 font-sans flex items-center gap-1 ml-auto group-hover:text-[#3B82F6]/60 transition-colors">
                          <ExternalLink className="w-2.5 h-2.5" />
                          Read
                        </span>
                      </div>
                      <h3 className="text-sm font-semibold font-sans text-foreground group-hover:text-[#3B82F6] transition-colors duration-300 line-clamp-2 mb-1.5 leading-snug">
                        {article.title}
                      </h3>
                      <WhyLine article={article} />
                      <div className="flex items-center justify-between gap-3 flex-wrap">
                        <CoverageRow article={article} k={k} />
                        <AnalyzeButton />
                      </div>
                    </div>
                  </div>
                </div>
              </a>
              <SourcesPanel article={article} k={k} />
            </div>
          );
        })}

        {/* Empty State */}
        {filtered.length === 0 && !loading && !error && (
          <div className={`${glass} rounded-2xl p-12 text-center`}>
            <Newspaper className="w-12 h-12 mx-auto mb-4 text-white/10" />
            <p className="text-sm font-sans text-muted-foreground">
              {articles.length ? 'No stories in this category' : 'No news articles available'}
            </p>
            <p className="text-xs text-muted-foreground/50 font-sans mt-1">
              The feed will automatically refresh
            </p>
          </div>
        )}
      </div>

      {/* Show More / Show Less */}
      {rest.length > 9 && (
        <div className="mt-5 text-center">
          <button
            onClick={() => setShowAll(!showAll)}
            className="inline-flex items-center gap-2 text-sm font-sans text-muted-foreground hover:text-foreground transition-all duration-300 px-5 py-2.5 border dark:border-white/[0.08] border-black/[0.08] rounded-full hover:dark:border-white/[0.15] hover:border-black/[0.15] hover:dark:bg-white/[0.04] hover:shadow-[0_0_15px_rgba(59,130,246,0.06)]"
          >
            {showAll ? (
              <>
                <ChevronUp className="w-3.5 h-3.5" />
                Show less
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5" />
                Show more ({Math.min(rest.length, 19) - 9} more)
              </>
            )}
          </button>
        </div>
      )}

      {/* Inline refresh indicator */}
      {loading && articles.length > 0 && (
        <div className="mt-3 flex items-center justify-center gap-2 text-xs text-muted-foreground/50">
          <RefreshCw className="w-3 h-3 animate-spin" />
          <span>Refreshing...</span>
        </div>
      )}

      <p className="mt-6 text-center text-[11px] text-muted-foreground/40 font-sans">
        Tags and summaries are generated by Gemma 2 from headlines only. Coverage counts show how many outlets carry a story, not whether it is true.
      </p>
    </div>
  );
}
