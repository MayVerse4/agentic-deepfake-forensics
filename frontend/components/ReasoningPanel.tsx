/**
 * ReasoningPanel.tsx -- renders the Phase 4A reasoning report returned by the
 * FastAPI /analyze endpoint as `result.reasoning`.
 *
 * Self-contained: plain React + Tailwind utility classes, no shadcn or other
 * imports, no hooks (works in a server or client component).
 *
 * Usage:
 *   <ReasoningPanel reasoning={result.reasoning} />
 */
import React from "react";

// ---- types: mirror ReasoningReport.to_dict() in agentic_reasoning.py ----
export type Verdict = "FAKE" | "REAL" | "INCONCLUSIVE";
export type Level = "Low" | "Medium" | "High";
export type Reliability = "reliable" | "limited" | "unreliable" | "unavailable";

export interface SignalAssessment {
  signal: "spatial" | "frequency" | "temporal";
  score: number | null; // P(fake), 0-1
  reliability: Reliability;
  contributes_to_verdict: boolean;
  separation: number;
  note: string;
}

export interface Caveat {
  code: string;
  severity: "warning" | "info";
  text: string;
}

export interface TrackReasoning {
  track_id: number | string | null;
  verdict: Verdict;
  model_verdict: "FAKE" | "REAL";
  confidence: Level;
  confidence_reasons: string[];
  fusion_prob_fake: number;
  signals: SignalAssessment[];
  caveats: Caveat[];
  quality_level: "good" | "degraded" | "poor" | "unknown";
  explanation: string;
  explanation_source: "llm" | "rule_based";
}

export interface ReasoningReportData {
  schema_version: string;
  input_type: "image" | "video";
  overall_verdict: Verdict;
  overall_confidence: Level;
  driving_track_id: number | string | null;
  tracks: TrackReasoning[];
  notes: Caveat[];
}

export interface ReasoningPanelProps {
  reasoning?: ReasoningReportData | { error: string } | null;
}

// ---- styling maps ----
const VERDICT_STYLE: Record<Verdict, string> = {
  FAKE: "bg-red-100 text-red-800 border-red-300 dark:bg-red-950 dark:text-red-200 dark:border-red-800",
  REAL: "bg-green-100 text-green-800 border-green-300 dark:bg-green-950 dark:text-green-200 dark:border-green-800",
  INCONCLUSIVE:
    "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-800",
};

const VERDICT_LABEL: Record<Verdict, string> = {
  FAKE: "Likely manipulated",
  REAL: "Likely authentic",
  INCONCLUSIVE: "Inconclusive",
};

const RELIABILITY_STYLE: Record<Reliability, string> = {
  reliable: "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-200",
  limited: "bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-200",
  unreliable: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200",
  unavailable: "bg-neutral-200 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300",
};

const SIGNAL_LABEL: Record<SignalAssessment["signal"], string> = {
  spatial: "Spatial",
  frequency: "Frequency",
  temporal: "Temporal",
};

const pct = (x: number | null) => (x === null ? "n/a" : `${Math.round(x * 100)}%`);

function Badge({ className, children }: { className: string; children: React.ReactNode }) {
  return <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${className}`}>{children}</span>;
}

function SignalCard({ s }: { s: SignalAssessment }) {
  return (
    <details className="rounded-lg border border-neutral-200 p-3 dark:border-neutral-800">
      <summary className="cursor-pointer list-none">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-semibold">{SIGNAL_LABEL[s.signal]}</span>
          <Badge className={RELIABILITY_STYLE[s.reliability]}>{s.reliability}</Badge>
        </div>
        <div className="mt-1 flex items-baseline justify-between text-xs text-neutral-600 dark:text-neutral-400">
          <span>P(fake): {pct(s.score)}</span>
          <span>{s.contributes_to_verdict ? "used in verdict" : "not used in verdict"}</span>
        </div>
      </summary>
      <p className="mt-2 text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">{s.note}</p>
    </details>
  );
}

function TrackCard({ t, showId }: { t: TrackReasoning; showId: boolean }) {
  const warnings = t.caveats.filter((c) => c.severity === "warning");
  const infos = t.caveats.filter((c) => c.severity === "info");

  return (
    <div className="space-y-4 rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
      <div className="flex flex-wrap items-center gap-2">
        {showId && <span className="text-sm font-semibold">Person {String(t.track_id)}</span>}
        <span className={`rounded border px-2 py-1 text-sm font-bold ${VERDICT_STYLE[t.verdict]}`}>
          {VERDICT_LABEL[t.verdict]}
        </span>
        <Badge className="bg-neutral-200 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-200">
          {t.confidence} confidence
        </Badge>
        <span className="text-xs text-neutral-500">Fusion score {t.fusion_prob_fake.toFixed(2)}</span>
      </div>

      {t.verdict === "INCONCLUSIVE" && t.model_verdict && (
        <p className="text-xs text-neutral-500">
          The raw model output was {t.model_verdict}, but the evidence is not strong enough to report it as a verdict.
        </p>
      )}

      <p className="text-sm leading-relaxed">{t.explanation}</p>
      <p className="text-xs text-neutral-500">
        {t.explanation_source === "llm" ? "Explanation written by an AI model." : "Explanation generated by rules."}{" "}
        Automated estimate, not proof.
      </p>

      <div className="grid gap-2 sm:grid-cols-3">
        {t.signals.map((s) => (
          <SignalCard key={s.signal} s={s} />
        ))}
      </div>

      {t.confidence_reasons.length > 0 && (
        <details className="text-xs text-neutral-600 dark:text-neutral-400">
          <summary className="cursor-pointer font-medium">Why this confidence level</summary>
          <ul className="mt-1 list-disc pl-5">
            {t.confidence_reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </details>
      )}

      {warnings.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold">Things to keep in mind</h4>
          <ul className="space-y-2">
            {warnings.map((c) => (
              <li
                key={c.code}
                className="rounded-lg border border-amber-300 bg-amber-50 p-2 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100"
              >
                {c.text}
              </li>
            ))}
          </ul>
        </div>
      )}

      {infos.length > 0 && (
        <details className="text-xs text-neutral-600 dark:text-neutral-400">
          <summary className="cursor-pointer font-medium">More context</summary>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {infos.map((c) => (
              <li key={c.code}>{c.text}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export default function ReasoningPanel({ reasoning }: ReasoningPanelProps) {
  if (!reasoning) return null;

  if ("error" in reasoning) {
    return (
      <div className="rounded-lg border border-neutral-300 p-3 text-sm text-neutral-600 dark:border-neutral-700 dark:text-neutral-400">
        Detailed reasoning is unavailable for this analysis.
      </div>
    );
  }

  const multi = reasoning.tracks.length > 1;

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-lg font-semibold">Analysis reasoning</h3>
        <span className={`rounded border px-2 py-1 text-sm font-bold ${VERDICT_STYLE[reasoning.overall_verdict]}`}>
          {VERDICT_LABEL[reasoning.overall_verdict]}
        </span>
        <Badge className="bg-neutral-200 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-200">
          {reasoning.overall_confidence} confidence
        </Badge>
      </div>

      {reasoning.notes.map((n) => (
        <div
          key={n.code}
          className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100"
        >
          {n.text}
        </div>
      ))}

      {reasoning.tracks.map((t, i) => (
        <TrackCard key={String(t.track_id) + i} t={t} showId={multi} />
      ))}
    </section>
  );
}