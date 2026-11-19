'use client';

/**
 * Media Analysis page. Talks directly to backend/forensics_api.py
 * (POST /jobs, poll GET /jobs/:id, POST /jobs/:id/ask).
 *
 * Every score comes from the trained models; the verdict comes from the
 * reasoning layer (agentic_reasoning.py); the local LLM only writes the
 * explanation and answers follow-up questions.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Upload, X, Loader2, Shield, ScanFace, Waves, Film, FileSearch, Bot, CheckCircle2,
  Circle, AlertTriangle, Download, MinusCircle, Clock, Sparkles, Send, MessageSquare, UserX,
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, ReferenceArea, ReferenceLine, CartesianGrid,
} from 'recharts';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { API } from '@/lib/api';


const BRANCHES = [
  { key: 'spatial', name: 'Spatial', model: 'EfficientNet-B0 · face crops', icon: ScanFace, color: '#3B82F6' },
  { key: 'frequency', name: 'Frequency', model: 'ResNet18 · FFT high-pass', icon: Waves, color: '#A855F7' },
  { key: 'temporal', name: 'Temporal', model: 'ResNet18 + LSTM · 16-frame', icon: Film, color: '#14B8A6' },
] as const;

const ASSESSMENT_LABEL: Record<string, string> = {
  ai_provenance: 'AI provenance declared',
  ai_tool_signature: 'AI generator signature',
  edited: 'Edited in software',
  screen_recording: 'Screen recording',
  c2pa_signed: 'Signed provenance',
  camera_original: 'Camera-original',
  re_encoded: 'Re-encoded',
  platform_reencoded: 'Messaging-app copy',
  stripped: 'Metadata stripped',
};

const PIPELINE = [
  { key: 'metadata', label: 'Metadata' },
  { key: 'branches', label: 'Detectors' },
  { key: 'reasoning', label: 'Agent reasoning' },
  { key: 'explaining', label: 'Report' },
];

interface Job {
  localId: string;
  file: File;
  previewUrl: string;
  jobId?: string;
  data?: any;
  error?: string;
  saved?: boolean;
}

function summarizeForSidebar(d: any): Record<string, string> {
  const v = d.verdict, b = d.branches || {}, m = d.media;
  const out: Record<string, string> = {};
  (['spatial', 'frequency', 'temporal'] as const).forEach((k) => {
    const x = b[k];
    out[k] = x?.status === 'skipped' ? 'Last: not applicable (image)'
      : x?.score !== undefined ? `Last: ${x.score.toFixed(0)}/100`
      : 'Last: n/a';
  });
  if (m) {
    const label = (ASSESSMENT_LABEL as any)[m.assessment] || m.assessment;
    out.metadata = `Last: ${label}`;
  }
  if (v) out.agent = v.not_assessed ? `Last: ${v.classification} (no face)` : `Last: ${v.classification} · ${v.final_score.toFixed(0)}/100`;
  return out;
}

function fmtT(t: number | null | undefined) {
  if (t === null || t === undefined) return 'n/a';
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

function verdictTone(cls?: string) {
  const c = (cls || '').toLowerCase();
  if (c.includes('inconclusive') || c.includes('possible edit') || c.includes('possibly ai')) return { text: 'text-amber-500', ring: '#F59E0B', bg: 'bg-amber-500/10 border-amber-500/30' };
  if (c.includes('edited region') || c.includes('edited (metadata')) return { text: 'text-red-500', ring: '#EF4444', bg: 'bg-red-500/10 border-red-500/30' };
  if (c.includes('cannot determine') || c.includes('no traces')) return { text: 'text-slate-400', ring: '#94A3B8', bg: 'bg-slate-500/10 border-slate-500/30' };
  if (c.includes('fake') || c.includes('ai-generated')) return { text: 'text-red-500', ring: '#EF4444', bg: 'bg-red-500/10 border-red-500/30' };
  if (c.includes('suspicious')) return { text: 'text-amber-500', ring: '#F59E0B', bg: 'bg-amber-500/10 border-amber-500/30' };
  return { text: 'text-emerald-500', ring: '#10B981', bg: 'bg-emerald-500/10 border-emerald-500/30' };
}

const panel = 'rounded-xl border dark:border-white/[0.07] border-black/[0.07] dark:bg-white/[0.02] bg-black/[0.02]';
const label = 'text-[10px] font-sans uppercase tracking-[0.18em] text-muted-foreground';

// ---------------------------------------------------------------------------
export default function ForensicsSection({ onActiveAgentsChange, onAgentSummaryChange }: { onActiveAgentsChange?: (ids: string[]) => void; onAgentSummaryChange?: (s: Record<string, string>) => void }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [health, setHealth] = useState<any>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let alive = true;
    const check = () => fetch(`${API}/health`).then((r) => r.json()).then((d) => alive && setHealth(d)).catch(() => alive && setHealth(null));
    check();
    const id = setInterval(check, 10000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const patch = useCallback((localId: string, p: Partial<Job>) => {
    setJobs((prev) => prev.map((j) => (j.localId === localId ? { ...j, ...p } : j)));
  }, []);

  // Light up the sidebar agents that are currently working
  useEffect(() => {
    if (!onActiveAgentsChange) return;
    const active = new Set<string>();
    jobs.forEach((j) => {
      const d = j.data;
      if (!d || d.status !== 'running') return;
      if (d.stage === 'metadata') active.add('metadata');
      if (d.stage === 'branches') Object.entries(d.branches || {}).forEach(([k, v]: any) => v.status === 'running' && active.add(k));
      if (d.stage === 'reasoning' || d.stage === 'explaining') active.add('agent');
    });
    onActiveAgentsChange(Array.from(active));
  }, [jobs, onActiveAgentsChange]);

  const start = useCallback(async (job: Job) => {
    const fd = new FormData();
    fd.append('file', job.file, job.file.name);
    let jobId: string;
    try {
      const res = await fetch(`${API}/jobs`, { method: 'POST', body: fd });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.detail || `HTTP ${res.status}`);
      jobId = body.job_id;
    } catch (e: any) {
      patch(job.localId, { error: `Could not reach the analysis backend (${e?.message || e}). Is backend/forensics_api.py running?` });
      return;
    }
    patch(job.localId, { jobId });
    const poll = async () => {
      try {
        const d = await fetch(`${API}/jobs/${jobId}`).then((r) => r.json());
        patch(job.localId, { data: d });
        if (d.status === 'done' || d.status === 'error') {
          if (d.status === 'error') patch(job.localId, { error: d.error });
          else onAgentSummaryChange?.(summarizeForSidebar(d));  // the backend saves it to History
          return;
        }
      } catch { /* transient */ }
      setTimeout(poll, 450);
    };
    poll();
  }, [patch]);

  const addFiles = useCallback((list: FileList | File[]) => {
    const accepted = Array.from(list).filter((f) => f.type.startsWith('image/') || f.type.startsWith('video/') || f.type.startsWith('audio/'));
    const newJobs: Job[] = accepted.map((file) => ({
      localId: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      file,
      previewUrl: URL.createObjectURL(file),
    }));
    setJobs((prev) => [...newJobs, ...prev]);
    newJobs.forEach(start);
  }, [start]);

  const remove = (id: string) => setJobs((prev) => {
    const j = prev.find((x) => x.localId === id);
    if (j) URL.revokeObjectURL(j.previewUrl);
    return prev.filter((x) => x.localId !== id);
  });

  const llm = health?.llm;

  return (
    <div className="w-full max-w-[1400px]">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
        <div>
          <h2 className="text-2xl font-bold font-sans tracking-tight text-foreground">Media Analysis</h2>
          <p className="text-sm text-muted-foreground font-sans mt-1">
            Multi-branch deepfake forensics · Spatial · Frequency · Temporal · Metadata · Agentic reasoning
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 h-9 rounded-xl border text-xs font-sans ${panel}`}>
            <span className={`w-2 h-2 rounded-full ${!health ? 'bg-red-500' : llm?.online && llm?.model_available ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]' : 'bg-amber-500'}`} />
            <span className="text-muted-foreground">
              {!health ? 'Backend offline' : llm?.online ? `Agent LLM · ${({ 'gemma2:latest': 'Gemma 2 9B', 'gemma2:9b': 'Gemma 2 9B', 'gemma2:2b': 'Gemma 2 2B' } as Record<string, string>)[llm.model] ?? llm.model}${llm.model_available === false ? ' (not downloaded)' : ''}` : 'Agent LLM offline'}
            </span>
          </div>
        </div>
      </div>

      {/* Upload */}
      <Card className="border dark:border-white/[0.08] border-black/[0.08] dark:bg-white/[0.03] bg-black/[0.03] backdrop-blur-xl rounded-2xl mb-6">
        <CardContent className="p-5">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files); }}
            onClick={() => inputRef.current?.click()}
            className={`flex items-center gap-5 border-2 border-dashed rounded-2xl px-6 py-7 cursor-pointer transition-all duration-300 ${dragOver ? 'border-[#3B82F6] bg-[#3B82F6]/10 shadow-[0_0_40px_rgba(59,130,246,0.15)]' : 'dark:border-white/[0.12] border-black/[0.12] hover:dark:border-white/[0.25] hover:border-black/[0.25]'}`}
          >
            <input ref={inputRef} type="file" accept="image/*,video/*,audio/*" multiple className="hidden"
              onChange={(e) => { if (e.target.files?.length) addFiles(e.target.files); e.target.value = ''; }} />
            <div className="w-14 h-14 shrink-0 rounded-2xl dark:bg-white/[0.05] bg-black/[0.05] border dark:border-white/[0.08] border-black/[0.08] flex items-center justify-center">
              <Upload className="w-6 h-6 text-muted-foreground" />
            </div>
            <div>
              <p className="text-sm font-sans text-foreground/85">Drop images or videos here, or click to browse</p>
              <p className="text-xs text-muted-foreground mt-1 font-sans">MP4 · MOV · AVI · MKV · JPG · PNG — multiple files supported</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {jobs.length === 0 ? (
        <Card className="border dark:border-white/[0.08] border-black/[0.08] dark:bg-white/[0.03] bg-black/[0.03] rounded-2xl">
          <CardContent className="p-14 text-center">
            <Shield className="w-10 h-10 mx-auto mb-4 text-muted-foreground/30" />
            <p className="text-sm font-sans text-muted-foreground">No analyses yet</p>
            <p className="text-xs text-muted-foreground/60 font-sans mt-1">Upload media to run the forensic pipeline</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {jobs.map((j) => <JobCard key={j.localId} job={j} onRemove={remove} />)}
        </div>
      )}

      <p className="text-xs text-muted-foreground/60 font-sans italic leading-relaxed mt-6">
        VerifAI provides probabilistic assessments. Results should be interpreted by qualified analysts and not used as sole evidence.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
function JobCard({ job, onRemove }: { job: Job; onRemove: (id: string) => void }) {
  const d = job.data;
  const videoRef = useRef<HTMLVideoElement>(null);
  const isVideo = job.file.type.startsWith('video');
  const running = !job.error && d?.status !== 'done';
  const seek = (t: number | null | undefined) => {
    if (t === null || t === undefined || !videoRef.current) return;
    videoRef.current.currentTime = t;
    videoRef.current.play().catch(() => {});
  };
  const media = d?.media;
  const ev = d?.evidence;

  return (
    <Card className="border dark:border-white/[0.08] border-black/[0.08] dark:bg-white/[0.03] bg-black/[0.03] backdrop-blur-xl rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-6 pt-5 pb-4 border-b dark:border-white/[0.06] border-black/[0.06]">
        <div className="min-w-0">
          <p className="text-sm font-sans font-medium text-foreground truncate">{job.file.name}</p>
          <p className="text-xs text-muted-foreground font-mono mt-0.5">
            {media ? [media.media_type, media.duration_s ? fmtT(media.duration_s) : null, media.resolution, media.fps ? `${media.fps} fps` : null].filter(Boolean).join(' · ') : 'Preparing…'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {running && (
            <span className="flex items-center gap-1.5 text-[11px] font-sans px-2.5 py-1 rounded-full bg-[#3B82F6]/15 text-[#3B82F6] border border-[#3B82F6]/25">
              <Loader2 className="w-3 h-3 animate-spin" /> {d?.elapsed ? `${d.elapsed.toFixed(0)}s` : 'Starting'}
            </span>
          )}
          <button onClick={() => onRemove(job.localId)} className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:dark:bg-white/[0.05] hover:bg-black/[0.05]">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <CardContent className="p-6 space-y-5">
        {job.error && (
          <div className="flex items-center gap-3 p-3 rounded-xl bg-red-500/10 border border-red-500/30">
            <AlertTriangle className="w-4 h-4 text-red-500 shrink-0" />
            <p className="text-sm text-red-500 font-sans">{job.error}</p>
          </div>
        )}

        {/* Preview + verdict */}
        <div className="grid lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] gap-5">
          <div className={`${panel} overflow-hidden flex items-center justify-center bg-black/80 min-h-[240px]`}>
            {isVideo
              ? <video ref={videoRef} src={job.previewUrl} controls muted className="w-full max-h-[340px] object-contain" />
              : job.file.type.startsWith('audio')
                ? <audio src={job.previewUrl} controls className="w-[90%]" />
                : <img src={job.previewUrl} alt={job.file.name} className="w-full max-h-[340px] object-contain" />}
          </div>
          <VerdictPanel d={d} error={!!job.error} />
        </div>

        {/* Branches side by side */}
        <div className="grid md:grid-cols-3 gap-4">
          {BRANCHES.map((b) => <BranchCard key={b.key} meta={b} state={d?.branches?.[b.key]} />)}
        </div>

        {/* Timeline */}
        {isVideo && ev?.timeline?.length > 0 && (
          <Timeline ev={ev} branches={Object.keys(d.branches || {}).filter((k) => d.branches[k].score !== undefined)} onSeek={seek} />
        )}

        {/* Evidence + metadata */}
        {(ev || media) && (
          <div className="grid md:grid-cols-2 gap-4">
            <EvidencePanel ev={ev} onSeek={seek} />
            <MetadataPanel media={media} />
          </div>
        )}

        {/* Pixel-level maps (images) */}
        {d?.forensics && <ForensicMaps fx={d.forensics} />}

        {/* Agent */}
        {d && <AgentPanel d={d} jobId={job.jobId} />}

        {d?.status === 'done' && (
          <Button onClick={() => downloadReport(job)} className="dark:bg-white/[0.06] bg-black/[0.06] hover:bg-white/[0.1] text-foreground border dark:border-white/[0.1] border-black/[0.1] font-sans rounded-xl text-xs h-9">
            <Download className="mr-2 w-3.5 h-3.5" /> Download report
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
function VerdictPanel({ d, error }: { d: any; error: boolean }) {
  const v = d?.verdict;
  const stageIdx = PIPELINE.findIndex((p) => p.key === d?.stage);
  const done = d?.status === 'done';

  if (!v) {
    return (
      <div className={`${panel} p-5 flex flex-col`}>
        <p className={label}>Pipeline</p>
        <div className="mt-4 space-y-3 flex-1">
          {PIPELINE.map((p, i) => {
            const state = error ? (i <= stageIdx ? 'error' : 'todo') : i < stageIdx ? 'done' : i === stageIdx ? 'active' : 'todo';
            return (
              <div key={p.key} className="flex items-center gap-3">
                {state === 'done' ? <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  : state === 'active' ? <Loader2 className="w-4 h-4 text-[#3B82F6] animate-spin" />
                  : state === 'error' ? <AlertTriangle className="w-4 h-4 text-red-500" />
                  : <Circle className="w-4 h-4 text-muted-foreground/40" />}
                <span className={`text-sm font-sans ${state === 'todo' ? 'text-muted-foreground/60' : 'text-foreground/85'}`}>{p.label}</span>
              </div>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground font-sans mt-4">Verdict appears once all detectors report.</p>
      </div>
    );
  }

  const tone = verdictTone(v.classification);
  const r = 46, c = 2 * Math.PI * r;
  if (v.not_assessed) {
    return (
      <div className={`${panel} p-5 flex flex-col`}>
        <p className={label}>Verdict</p>
        <div className="flex items-center gap-5 mt-3">
          <div className="w-[120px] h-[120px] shrink-0 rounded-full border-[10px] border-dashed border-slate-500/30 flex items-center justify-center">
            <UserX className="w-9 h-9 text-slate-400" />
          </div>
          <div className="min-w-0">
            <span className={`inline-block text-sm font-bold font-sans px-3 py-1 rounded-full border ${tone.bg} ${tone.text}`}>Cannot Determine</span>
            <p className="text-sm font-sans font-medium text-foreground mt-3">No face detected</p>
            <p className="text-xs font-sans text-muted-foreground mt-1 leading-relaxed">
              The detectors analyse faces only, so the system cannot determine whether this is fake or real.
            </p>
          </div>
        </div>
        {!done && (
          <p className="text-xs text-muted-foreground font-sans mt-auto pt-4 flex items-center gap-2">
            <Loader2 className="w-3 h-3 animate-spin" /> Agent is writing the report…
          </p>
        )}
      </div>
    );
  }
  return (
    <div className={`${panel} p-5 flex flex-col`}>
      <p className={label}>Verdict</p>
      <div className="flex items-center gap-6 mt-3">
        <svg width="120" height="120" viewBox="0 0 120 120" className="shrink-0">
          <circle cx="60" cy="60" r={r} fill="none" strokeWidth="10" className="stroke-black/[0.07] dark:stroke-white/[0.08]" />
          <circle cx="60" cy="60" r={r} fill="none" strokeWidth="10" stroke={tone.ring} strokeLinecap="round"
            strokeDasharray={c} strokeDashoffset={c * (1 - v.final_score / 100)} transform="rotate(-90 60 60)"
            style={{ transition: 'stroke-dashoffset 900ms ease-out', filter: `drop-shadow(0 0 6px ${tone.ring}66)` }} />
          <text x="60" y="58" textAnchor="middle" className="fill-foreground" style={{ font: '700 26px ui-monospace, monospace' }}>{v.final_score.toFixed(0)}</text>
          <text x="60" y="78" textAnchor="middle" className="fill-muted-foreground" style={{ font: '10px ui-sans-serif' }}>/ 100 manip.</text>
        </svg>
        <div className="min-w-0">
          <span className={`inline-block text-sm font-bold font-sans px-3 py-1 rounded-full border ${tone.bg} ${tone.text}`}>{v.classification}</span>
          <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs font-sans">
            <dt className="text-muted-foreground">Confidence</dt><dd className="text-foreground font-medium">{v.confidence_level}</dd>
            <dt className="text-muted-foreground">Top signal</dt><dd className="text-foreground font-medium">{v.top_contributing_signal}</dd>
            <dt className="text-muted-foreground">Branch spread</dt><dd className="text-foreground font-medium">{v.branch_spread.toFixed(0)} pts</dd>
            {v.override_applied && (<><dt className="text-muted-foreground">Override</dt><dd className="text-red-500 font-medium">Metadata provenance</dd></>)}
          </dl>
        </div>
      </div>
      {!done && (
        <p className="text-xs text-muted-foreground font-sans mt-auto pt-4 flex items-center gap-2">
          <Loader2 className="w-3 h-3 animate-spin" /> Agent is writing the report…
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
function BranchCard({ meta, state }: { meta: typeof BRANCHES[number]; state: any }) {
  const Icon = meta.icon;
  const status = state?.status || 'queued';
  const score: number | undefined = state?.score;
  return (
    <div className={`${panel} p-4 relative overflow-hidden`}>
      <div className="absolute inset-x-0 top-0 h-[2px]" style={{ background: status === 'skipped' ? 'transparent' : meta.color, opacity: status === 'done' ? 0.9 : 0.35 }} />
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${meta.color}1f` }}>
            <Icon className="w-4 h-4" style={{ color: meta.color }} />
          </div>
          <div>
            <p className="text-sm font-sans font-semibold text-foreground">{meta.name}</p>
            <p className="text-[11px] font-sans text-muted-foreground">{meta.model}</p>
          </div>
        </div>
        {status === 'done' && score !== undefined
          ? <span className="text-2xl font-bold font-mono text-foreground">{score.toFixed(0)}</span>
          : status === 'running' ? <Loader2 className="w-4 h-4 animate-spin" style={{ color: meta.color }} />
          : status === 'skipped' ? <MinusCircle className="w-4 h-4 text-muted-foreground/50" />
          : <Clock className="w-4 h-4 text-muted-foreground/40" />}
      </div>

      <div className="mt-4 h-1.5 w-full rounded-full dark:bg-white/[0.06] bg-black/[0.06] overflow-hidden">
        {status === 'running' && state?.progress === null ? (
          <div className="h-full w-1/3 rounded-full animate-[indeterminate_1.2s_ease-in-out_infinite]" style={{ background: meta.color }} />
        ) : (
          <div className="h-full rounded-full transition-all duration-300 ease-out"
            style={{ width: `${status === 'done' && score !== undefined ? score : state?.progress ?? 0}%`, background: meta.color, boxShadow: `0 0 8px ${meta.color}88` }} />
        )}
      </div>
      <p className="text-xs font-sans mt-2.5 min-h-[32px] leading-relaxed text-muted-foreground">
        {status === 'done' && state?.finding ? <span className="text-foreground/80">{state.finding}</span>
          : status === 'running' ? `${state.stage}${state.progress != null ? ` · ${state.progress}%` : '…'}`
          : state?.stage || 'Waiting'}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
function Timeline({ ev, branches, onSeek }: { ev: any; branches: string[]; onSeek: (t: number) => void }) {
  return (
    <div className={`${panel} p-4`}>
      <div className="flex items-center justify-between mb-3">
        <p className={label}>Manipulation score over time</p>
        <div className="flex items-center gap-4">
          {BRANCHES.filter((b) => branches.includes(b.key)).map((b) => (
            <span key={b.key} className="flex items-center gap-1.5 text-xs font-sans text-muted-foreground">
              <span className="w-3 h-[3px] rounded-full" style={{ background: b.color }} />{b.name}
            </span>
          ))}
          <span className="flex items-center gap-1.5 text-xs font-sans text-muted-foreground">
            <span className="w-3 h-3 rounded-sm bg-red-500/20 border border-red-500/40" />Flagged
          </span>
        </div>
      </div>
      <div className="h-[200px] text-muted-foreground">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={ev.timeline} margin={{ top: 5, right: 8, left: -20, bottom: 0 }}
            onClick={(e: any) => e?.activeLabel !== undefined && onSeek(Number(e.activeLabel))}>
            <CartesianGrid stroke="currentColor" strokeOpacity={0.08} vertical={false} />
            <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(t) => fmtT(t)} stroke="currentColor" strokeOpacity={0.4} fontSize={11} tickLine={false} />
            <YAxis domain={[0, 100]} ticks={[0, 50, 100]} stroke="currentColor" strokeOpacity={0.4} fontSize={11} tickLine={false} axisLine={false} />
            <ReferenceLine y={50} stroke="currentColor" strokeOpacity={0.25} strokeDasharray="4 4" />
            {ev.segments?.map((s: any, i: number) => (
              <ReferenceArea key={i} x1={s.start} x2={s.end} fill="#EF4444" fillOpacity={0.14} stroke="#EF4444" strokeOpacity={0.35} />
            ))}
            <Tooltip
              contentStyle={{ background: 'rgba(12,12,14,0.92)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 }}
              labelFormatter={(t) => `t = ${fmtT(Number(t))}`}
              formatter={(v: any, n: any) => [Number(v).toFixed(0), String(n).charAt(0).toUpperCase() + String(n).slice(1)]}
            />
            {BRANCHES.filter((b) => branches.includes(b.key)).map((b) => (
              <Line key={b.key} type="monotone" dataKey={b.key} stroke={b.color} strokeWidth={2} dot={false} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {ev.segments?.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {ev.segments.map((s: any, i: number) => (
            <button key={i} onClick={() => onSeek(s.start)}
              className="text-left text-xs font-sans px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/25 hover:bg-red-500/15 transition-colors">
              <span className="font-mono font-semibold text-red-500">{fmtT(s.start)}–{fmtT(s.end)}</span>
              <span className="text-muted-foreground"> · {s.branches.map((b: string) => b[0].toUpperCase() + b.slice(1)).join(', ')} · </span>
              <span className="text-foreground/80">{s.cues[0]}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
function ForensicMaps({ fx }: { fx: any }) {
  const loc = (fx.maps || []).find((m: any) => m.key === 'trufor');
  const others = (fx.maps || []).filter((m: any) => m.key !== 'trufor');
  const L = fx.localization;
  const AI = fx.ai;
  return (
    <div className="space-y-4">
    {AI && (
      <div className={`${panel} p-4`}>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <p className={label}>Whole-image AI-generation check</p>
          <span className="text-[10px] font-sans text-muted-foreground">Community Forensics · pixels only, works with metadata removed</span>
        </div>
        <div className="mt-3 flex items-center gap-4">
          <div className="flex-1 h-2.5 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
            <div className={`h-full ${AI.prob_ai >= 0.85 ? 'bg-red-500' : AI.prob_ai >= 0.5 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                 style={{ width: `${Math.max(2, AI.prob_ai * 100)}%` }} />
          </div>
          <span className="text-sm font-mono font-semibold text-foreground w-24 text-right">{(AI.prob_ai * 100).toFixed(0)}% AI</span>
        </div>
        <p className="text-[11px] font-sans text-muted-foreground leading-snug mt-2">
          Probability that the entire picture was produced by an AI generator{AI.frames ? ` (median over ${AI.frames} video frames)` : ''}.
          Face swaps inside real footage are a different case, handled by the face models. Less reliable on the newest generators.
        </p>
      </div>
    )}
    {loc && (
      <div className={`${panel} p-4`}>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <p className={label}>Manipulated-region localization</p>
          <span className="text-[10px] font-sans text-muted-foreground">TruFor · works on any region, from pixels only (metadata not needed)</span>
        </div>
        <div className="grid md:grid-cols-[2fr_1fr] gap-4 mt-3 items-start">
          <div className="rounded-lg overflow-hidden border dark:border-white/[0.08] border-black/[0.08] bg-black/60 flex items-center justify-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            {loc.image && <img src={loc.image} alt={loc.title} className="w-full max-h-[420px] object-contain" />}
          </div>
          <div className="space-y-2 text-xs font-sans">
            {L && (
              <>
                <p className="text-foreground"><span className="font-semibold">Manipulation score:</span> {(L.score * 100).toFixed(0)}/100</p>
                <p className="text-foreground"><span className="font-semibold">Reliability:</span> {L.reliability != null ? (L.reliability * 100).toFixed(0) + '%' : 'n/a'}</p>
                <p className="text-foreground"><span className="font-semibold">Suspected regions:</span> {L.score >= 0.5 ? (L.regions || []).length : 0}</p>
              </>
            )}
            <p className="text-[11px] text-muted-foreground leading-snug">{loc.caption}</p>
            <p className="text-[11px] text-muted-foreground leading-snug">Finds edited <em>parts</em> of an image. A fully AI-generated image has no edited part, so a low score does not prove authenticity.</p>
          </div>
        </div>
      </div>
    )}
    {others.length > 0 && (
    <div className={`${panel} p-4`}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className={label}>Image forensic maps</p>
        <span className="text-[10px] font-sans text-muted-foreground">Measured from the uploaded pixels · descriptive, not part of the score</span>
      </div>
      <div className="grid md:grid-cols-3 gap-3 mt-3">
        {others.map((m: any) => (
          <div key={m.key}>
            <div className="rounded-lg overflow-hidden border dark:border-white/[0.08] border-black/[0.08] bg-black/60 flex items-center justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              {m.image && <img src={m.image} alt={m.title} className="w-full max-h-[200px] object-contain" />}
            </div>
            <p className="text-xs font-sans font-semibold text-foreground mt-2">{m.title}</p>
            <p className="text-[11px] font-sans text-muted-foreground leading-snug mt-0.5">{m.caption}</p>
          </div>
        ))}
      </div>
      {(fx.measurements || []).length > 0 && (
        <ul className="mt-3 space-y-1">
          {fx.measurements.map((t: string, i: number) => (
            <li key={i} className="text-[11px] font-mono text-muted-foreground">{t}</li>
          ))}
        </ul>
      )}
    </div>
    )}
    </div>
  );
}

// ---------------------------------------------------------------------------
function EvidencePanel({ ev, onSeek }: { ev: any; onSeek: (t: number) => void }) {
  const frames = ev?.evidence || [];
  return (
    <div className={`${panel} p-4`}>
      <p className={label}>Evidence frames</p>
      {frames.length === 0 ? (
        <p className="text-xs text-muted-foreground font-sans mt-3">{ev ? 'No evidence frames extracted.' : 'Waiting for detectors…'}</p>
      ) : (
        <div className="grid grid-cols-2 gap-3 mt-3">
          {frames.map((f: any, i: number) => (
            <button key={i} onClick={() => onSeek(f.t)} className="text-left group">
              <div className={`relative rounded-lg overflow-hidden border ${f.flagged ? 'border-red-500/40' : 'border-emerald-500/30'}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={f.image} alt="" className="w-full aspect-video object-cover group-hover:scale-[1.02] transition-transform" />
                {f.t !== null && <span className="absolute top-1.5 left-1.5 text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/70 text-white">{fmtT(f.t)}</span>}
              </div>
              <p className={`text-[11px] font-sans mt-1.5 leading-snug ${f.flagged ? 'text-red-500/90' : 'text-emerald-600 dark:text-emerald-400'}`}>{f.note}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
const LEVEL_DOT: Record<string, string> = { high: 'bg-red-500', medium: 'bg-amber-500', low: 'bg-emerald-500', info: 'bg-muted-foreground/40' };

function MetadataPanel({ media }: { media: any }) {
  if (!media) return <div className={`${panel} p-4`}><p className={label}>Metadata & provenance</p><p className="text-xs text-muted-foreground mt-3">Reading container…</p></div>;
  const f = media.fields || {};
  const rows: [string, any][] = [
    ['C2PA credentials', media.c2pa?.present ? (media.c2pa.claim_generator || 'present') : 'none'],
    ['Device', media.device?.label ? `${media.device.label}${media.device.class ? ` · ${media.device.class}` : ''}` : '—'],
    ['OS', media.device?.os || '—'],
    ['Software', media.software || (media.platform ? `${media.platform} re-encoder` : '—')],
    ['Created', media.created || '—'],
    ['Container / codec', [f.major_brand?.trim(), media.codec || media.format].filter(Boolean).join(' / ') || '—'],
    ['SHA-256', media.sha256?.slice(0, 16) + '…'],
  ];
  return (
    <div className={`${panel} p-4`}>
      <div className="flex items-center justify-between">
        <p className={label}>Metadata & provenance</p>
        <span className="flex items-center gap-1.5 text-[11px] font-sans font-medium text-foreground/80">
          <FileSearch className="w-3.5 h-3.5 text-muted-foreground" />{ASSESSMENT_LABEL[media.assessment] || media.assessment}
        </span>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 mt-3 text-xs font-sans">
        {rows.map(([k, v]) => (<React.Fragment key={k}><dt className="text-muted-foreground">{k}</dt><dd className="text-foreground/85 font-mono truncate">{String(v)}</dd></React.Fragment>))}
      </dl>
      <ul className="mt-3 space-y-1.5 pt-3 border-t dark:border-white/[0.06] border-black/[0.06]">
        {media.findings.map((x: any, i: number) => (
          <li key={i} className="flex gap-2 text-xs font-sans text-foreground/75 leading-relaxed">
            <span className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${LEVEL_DOT[x.level]}`} />{x.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
function AgentPanel({ d, jobId }: { d: any; jobId?: string }) {
  const steps = d.steps || [];
  const writing = d.stage === 'explaining';
  const src: string = d.explanation_source || '';
  return (
    <div className="rounded-xl border border-[#3B82F6]/25 bg-gradient-to-br from-[#3B82F6]/[0.06] to-transparent p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[#3B82F6]/15 flex items-center justify-center"><Bot className="w-4 h-4 text-[#3B82F6]" /></div>
          <div>
            <p className="text-sm font-sans font-semibold text-foreground">Forensic Agent</p>
            <p className="text-[11px] font-sans text-muted-foreground">Cross-checks evidence · decides by rules · Gemma 2 9B (local, Ollama) explains</p>
          </div>
        </div>
        {d.status === 'running' && <Loader2 className="w-4 h-4 text-[#3B82F6] animate-spin" />}
      </div>
      <div className="grid lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] gap-5">
        <ol className="relative space-y-3 pl-5 before:absolute before:left-[5px] before:top-1.5 before:bottom-1.5 before:w-px before:bg-[#3B82F6]/25">
          {steps.map((s: any, i: number) => (
            <li key={i} className="relative">
              <span className={`absolute -left-5 top-1 w-[11px] h-[11px] rounded-full border-2 ${s.status === 'running' ? 'border-[#3B82F6] bg-background animate-pulse' : 'border-[#3B82F6] bg-[#3B82F6]'}`} />
              <p className="text-xs font-sans"><span className="font-semibold text-foreground">{s.title}</span><span className="text-muted-foreground font-mono ml-2">+{s.t}s</span></p>
              <p className="text-xs font-sans text-muted-foreground leading-relaxed mt-0.5">{s.detail}</p>
            </li>
          ))}
          {steps.length === 0 && <li className="text-xs text-muted-foreground">Starting…</li>}
        </ol>
        <div className={`${panel} p-4`}>
          <p className={`${label} flex items-center gap-1.5`}><Sparkles className="w-3 h-3" /> Analyst explanation</p>
          <p className="text-sm font-sans text-foreground/85 leading-relaxed mt-3 whitespace-pre-wrap">
            {d.explanation || (d.stage === 'explaining' ? '' : <span className="text-muted-foreground text-xs">Written once the verdict is reached.</span>)}
            {writing && <span className="inline-block w-[7px] h-[15px] ml-0.5 align-[-2px] bg-[#3B82F6] animate-pulse" />}
          </p>
          {src && <p className="text-[10px] font-mono text-muted-foreground/70 mt-3">generated by {src === 'gemma2:latest' ? 'Gemma 2 9B · local' : src}</p>}
        </div>
      </div>
      {d.status === 'done' && jobId && <AskAgent jobId={jobId} isVideo={d.media?.media_type === 'video'} hasSegments={(d.evidence?.segments || []).length > 0} firstSegment={d.evidence?.segments?.[0]} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
function AskAgent({ jobId, isVideo, hasSegments, firstSegment }: { jobId: string; isVideo: boolean; hasSegments: boolean; firstSegment?: any }) {
  const [msgs, setMsgs] = useState<{ q: string; a: string; done: boolean }[]>([]);
  const [input, setInput] = useState('');
  const busy = msgs.some((m) => !m.done);
  const endRef = useRef<HTMLDivElement>(null);

  const suggestions = [
    hasSegments && firstSegment ? `Why did you flag ${fmtT(firstSegment.start)} to ${fmtT(firstSegment.end)}?` : null,
    'Which detector do you trust most here, and why?',
    'What would change your verdict?',
    'Does the metadata agree with the detectors?',
    !isVideo ? 'Why was the temporal detector not used?' : null,
  ].filter(Boolean) as string[];

  const ask = async (question: string) => {
    const q = question.trim();
    if (!q || busy) return;
    setInput('');
    setMsgs((m) => [...m, { q, a: '', done: false }]);
    const update = (fn: (a: string) => string, done = false) =>
      setMsgs((m) => m.map((x, i) => (i === m.length - 1 ? { ...x, a: fn(x.a), done } : x)));
    try {
      const res = await fetch(`${API}/jobs/${jobId}/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        update((a) => a + chunk);
      }
      update((a) => a.trim(), true);
    } catch (e: any) {
      update(() => `Could not reach the agent (${e?.message || e}).`, true);
    }
  };

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [msgs]);

  return (
    <div className="mt-5 pt-5 border-t border-[#3B82F6]/20">
      <p className={`${label} flex items-center gap-1.5 mb-3`}><MessageSquare className="w-3 h-3" /> Ask the agent about this result</p>
      {msgs.length > 0 && (
        <div className="space-y-3 mb-4 max-h-[360px] overflow-y-auto pr-1">
          {msgs.map((m, i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-end">
                <p className="max-w-[80%] text-sm font-sans px-3.5 py-2 rounded-2xl rounded-br-md bg-[#3B82F6] text-white">{m.q}</p>
              </div>
              <div className="flex gap-2.5">
                <div className="w-7 h-7 shrink-0 rounded-lg bg-[#3B82F6]/15 flex items-center justify-center"><Bot className="w-3.5 h-3.5 text-[#3B82F6]" /></div>
                <p className={`max-w-[85%] text-sm font-sans leading-relaxed px-3.5 py-2 rounded-2xl rounded-tl-md ${panel} text-foreground/85`}>
                  {m.a || (!m.done && <span className="text-muted-foreground text-xs">Reviewing the evidence…</span>)}
                  {!m.done && m.a && <span className="inline-block w-[6px] h-[13px] ml-0.5 align-[-2px] bg-[#3B82F6] animate-pulse" />}
                </p>
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>
      )}
      {msgs.length === 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {suggestions.map((s) => (
            <button key={s} onClick={() => ask(s)}
              className="text-xs font-sans px-3 py-1.5 rounded-full border border-[#3B82F6]/30 text-[#3B82F6] hover:bg-[#3B82F6]/10 transition-colors">{s}</button>
          ))}
        </div>
      )}
      <form onSubmit={(e) => { e.preventDefault(); ask(input); }} className="flex gap-2">
        <input value={input} onChange={(e) => setInput(e.target.value)} disabled={busy}
          placeholder={busy ? 'Agent is answering…' : 'e.g. Why is the frequency score lower than spatial?'}
          className={`flex-1 h-10 px-3.5 text-sm font-sans rounded-xl outline-none focus:border-[#3B82F6]/60 ${panel} text-foreground placeholder:text-muted-foreground/60`} />
        <button type="submit" disabled={busy || !input.trim()}
          className="h-10 px-4 rounded-xl bg-[#3B82F6] text-white disabled:opacity-40 flex items-center gap-1.5 text-sm font-sans">
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        </button>
      </form>
    </div>
  );
}

async function downloadReport(job: Job) {
  const { jsPDF } = await import('jspdf');
  const d = job.data;
  const v = d.verdict;
  const media = d.media;
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const marginX = 48;
  const contentW = pageW - marginX * 2;
  let y = 0;

  const ink = { text: [23, 23, 23] as [number, number, number], muted: [110, 110, 118] as [number, number, number], line: [222, 222, 226] as [number, number, number] };
  const tone: [number, number, number] = v.not_assessed ? [110, 118, 130] : /suspicious|possible edit|possibly ai|inconclusive/.test(v.classification?.toLowerCase() || '') ? [196, 138, 20]
    : /fake|edited region|ai-generated/.test(v.classification?.toLowerCase() || '') ? [214, 60, 60]
    : /no traces/.test(v.classification?.toLowerCase() || '') ? [110, 118, 130] : [30, 140, 90];

  const ensureSpace = (h: number) => { if (y + h > pageH - 56) { addFooter(); doc.addPage(); y = 56; } };
  const rule = () => { doc.setDrawColor(...ink.line); doc.setLineWidth(0.75); doc.line(marginX, y, pageW - marginX, y); y += 18; };
  const heading = (text: string) => {
    ensureSpace(28);
    doc.setFont('helvetica', 'bold').setFontSize(11).setTextColor(...ink.text);
    doc.text(text.toUpperCase(), marginX, y);
    y += 6;
    doc.setDrawColor(...tone).setLineWidth(1.5).line(marginX, y, marginX + 28, y);
    y += 16;
  };
  const paragraph = (text: string, opts: { size?: number; color?: [number, number, number]; bold?: boolean; lineH?: number } = {}) => {
    const { size = 10, color = ink.text, bold = false, lineH = 14 } = opts;
    doc.setFont('helvetica', bold ? 'bold' : 'normal').setFontSize(size).setTextColor(...color);
    const lines = doc.splitTextToSize(text, contentW) as string[];
    lines.forEach((ln) => { ensureSpace(lineH); doc.text(ln, marginX, y); y += lineH; });
  };
  const bullet = (text: string, dotColor: [number, number, number] = ink.muted) => {
    doc.setFont('helvetica', 'normal').setFontSize(9.5).setTextColor(...ink.text);
    const lines = doc.splitTextToSize(text, contentW - 16) as string[];
    ensureSpace(lines.length * 13 + 4);
    doc.setFillColor(...dotColor);
    doc.circle(marginX + 3, y - 3, 2, 'F');
    lines.forEach((ln, i) => { doc.text(ln, marginX + 14, y); y += 13; });
    y += 3;
  };
  let pageNum = 1;
  const addFooter = () => {
    doc.setFont('helvetica', 'normal').setFontSize(8).setTextColor(...ink.muted);
    doc.text('VerifAI provides probabilistic assessments. Not to be used as sole evidence.', marginX, pageH - 34);
    doc.text(`Page ${pageNum}`, pageW - marginX, pageH - 34, { align: 'right' });
    pageNum += 1;
  };

  // ---- header ----
  y = 56;
  doc.setFont('helvetica', 'bold').setFontSize(18).setTextColor(...ink.text);
  doc.text('VerifAI Forensic Report', marginX, y);
  y += 22;
  doc.setFont('helvetica', 'normal').setFontSize(9.5).setTextColor(...ink.muted);
  doc.text(`${job.file.name}  ·  Generated ${new Date().toLocaleString()}`, marginX, y);
  y += 26;
  rule();

  // ---- verdict block ----
  ensureSpace(70);
  doc.setFillColor(tone[0], tone[1], tone[2]);
  doc.roundedRect(marginX, y, contentW, 58, 6, 6, 'F');
  doc.setFont('helvetica', 'bold').setFontSize(22).setTextColor(255, 255, 255);
  doc.text(v.not_assessed ? 'N/A' : `${Math.round(v.final_score)}/100`, marginX + 16, y + 34);
  doc.setFont('helvetica', 'bold').setFontSize(13);
  doc.text(v.classification || '', marginX + 130, y + 26);
  doc.setFont('helvetica', 'normal').setFontSize(9.5);
  if (v.not_assessed) doc.text('No face detected: the system cannot determine whether this is fake or real.', marginX + 130, y + 44);
  else doc.text(`Confidence: ${v.confidence_level}   ·   Top signal: ${v.top_contributing_signal}   ·   Branch spread: ${v.branch_spread} pts`, marginX + 130, y + 44);
  y += 78;

  // ---- detectors ----
  heading('Detectors');
  const cols = Object.entries(d.branches).filter(([, x]: any) => x.score !== undefined) as [string, any][];
  const colW = contentW / Math.max(cols.length, 1);
  ensureSpace(56);
  const rowY = y;
  cols.forEach(([k, x]: any, i) => {
    const cx = marginX + i * colW;
    doc.setFont('helvetica', 'bold').setFontSize(9).setTextColor(...ink.muted);
    doc.text(k.toUpperCase(), cx, rowY);
    doc.setFont('helvetica', 'bold').setFontSize(17).setTextColor(...ink.text);
    doc.text(String(Math.round(x.score)), cx, rowY + 22);
    doc.setFont('helvetica', 'normal').setFontSize(8.5).setTextColor(...ink.muted);
    const lines = doc.splitTextToSize(x.finding || x.stage || '', colW - 12) as string[];
    lines.slice(0, 2).forEach((ln, li) => doc.text(ln, cx, rowY + 36 + li * 11));
  });
  const skipped = Object.entries(d.branches).filter(([, x]: any) => x.status === 'skipped') as [string, any][];
  y = rowY + 56;
  skipped.forEach(([k, x]: any) => paragraph(`${k}: not used (${x.stage})`, { size: 8.5, color: ink.muted, lineH: 12 }));
  y += 6;

  // ---- suspicious segments ----
  if ((d.evidence?.segments || []).length) {
    heading('Suspicious segments');
    d.evidence.segments.forEach((s: any) => {
      bullet(`${fmtT(s.start)}-${fmtT(s.end)}  ·  ${s.branches.map((b: string) => b[0].toUpperCase() + b.slice(1)).join(', ')}  ·  ${s.cues.join('; ')}`, tone);
    });
    y += 6;
  }

  // ---- metadata ----
  heading('Metadata & provenance');
  const metaRows: [string, string][] = [
    ['Device', media?.device?.label ? `${media.device.label}${media.device.class ? ` (${media.device.class})` : ''}` : '—'],
    ['OS', media?.device?.os || '—'],
    ['Software', media?.software || (media?.platform ? `${media.platform} re-encoder` : '—')],
    ['Created', media?.created || '—'],
    ['Container / codec', [media?.fields?.major_brand?.trim(), media?.codec || media?.format].filter(Boolean).join(' / ') || '—'],
    ['C2PA credentials', media?.c2pa?.present ? (media.c2pa.claim_generator || 'present') : 'none'],
    ['SHA-256', media?.sha256 || '—'],
  ];
  metaRows.forEach(([k, val]) => {
    ensureSpace(13);
    doc.setFont('helvetica', 'bold').setFontSize(9).setTextColor(...ink.muted);
    doc.text(k, marginX, y);
    doc.setFont('helvetica', 'normal').setFontSize(9).setTextColor(...ink.text);
    doc.text(String(val), marginX + 120, y);
    y += 13;
  });
  y += 8;
  (media?.findings || []).forEach((f: any) => bullet(f.text, f.level === 'high' ? tone : ink.muted));
  y += 6;

  // ---- agent reasoning ----
  heading('Agent reasoning');
  paragraph(`Verdict decided by the reasoning layer; explanation by ${d.explanation_source || 'n/a'}`, { size: 8.5, color: ink.muted, lineH: 12 });
  y += 4;
  d.steps.forEach((s: any) => {
    paragraph(`${s.title}  +${s.t}s`, { size: 9.5, bold: true, lineH: 13 });
    paragraph(s.detail, { size: 9, color: ink.muted, lineH: 12 });
    y += 3;
  });
  y += 6;

  // ---- explanation ----
  heading('Analyst explanation');
  ensureSpace(20);
  doc.setDrawColor(...ink.line).setFillColor(248, 248, 249);
  const expLines = doc.splitTextToSize(d.explanation || '(no explanation generated)', contentW - 24) as string[];
  const boxH = expLines.length * 13 + 20;
  ensureSpace(boxH);
  doc.roundedRect(marginX, y, contentW, boxH, 4, 4, 'FD');
  doc.setFont('helvetica', 'normal').setFontSize(9.5).setTextColor(...ink.text);
  expLines.forEach((ln, i) => doc.text(ln, marginX + 12, y + 18 + i * 13));
  y += boxH + 10;

  addFooter();
  doc.save(`verifai-report-${job.file.name.replace(/\.[^/.]+$/, '')}.pdf`);
}
