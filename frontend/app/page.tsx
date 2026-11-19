'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LogOut, Loader2 } from 'lucide-react';
import { API, type SessionUser } from '@/lib/api';
import Sidebar from './sections/Sidebar';
import ForensicsSection from './sections/ForensicsSection';
import TrendingPage from './sections/TrendingPage';
import HistoryPage, { type AnalysisRecord } from './sections/HistoryPage';
import AuthScreen from './sections/AuthScreen';
import LandingPage from './sections/LandingPage';
import ThemeToggle from '@/components/ThemeToggle';

const AGENTS = [
  { id: 'agent', name: 'Forensic Agent', purpose: 'Cross-checks evidence, decides the verdict · Gemma 2 9B explains' },
  { id: 'spatial', name: 'Spatial Analysis', purpose: 'EfficientNet-B0 on tracked face crops' },
  { id: 'frequency', name: 'Frequency Analysis', purpose: 'ResNet18 on FFT high-pass residuals' },
  { id: 'temporal', name: 'Temporal Analysis', purpose: 'ResNet18 + LSTM over 16-frame sequences' },
  { id: 'metadata', name: 'Metadata & Provenance', purpose: 'C2PA credentials, container atoms, EXIF' },
];

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: '' };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
          <div className="text-center p-8 max-w-md bg-white/[0.03] backdrop-blur-xl border border-white/[0.08] rounded-2xl shadow-[0_0_30px_rgba(59,130,246,0.1)]">
            <h2 className="text-xl font-semibold mb-2 font-sans">Something went wrong</h2>
            <p className="text-[#9A9AA0] mb-4 text-sm font-sans">{this.state.error}</p>
            <button
              onClick={() => this.setState({ hasError: false, error: '' })}
              className="px-5 py-2.5 bg-gradient-to-r from-[#3B82F6] to-[#1D4ED8] text-white text-sm font-sans rounded-xl hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all duration-300"
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function ProfileMenu({ user, onLogout }: { user: SessionUser; onLogout: () => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const initial = user?.email?.charAt(0)?.toUpperCase() || 'U';

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="w-10 h-10 rounded-xl bg-gradient-to-r from-[#3B82F6] to-[#1D4ED8] flex items-center justify-center text-white font-bold text-sm hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all duration-300"
      >
        {initial}
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-56 bg-[#0D0D0F]/95 backdrop-blur-xl border border-white/[0.08] rounded-2xl shadow-[0_0_30px_rgba(0,0,0,0.5)] z-50 py-2 overflow-hidden">
          <div className="px-4 py-2.5 border-b border-white/[0.06]">
            {user.name && <p className="text-sm text-white/90 font-sans truncate">{user.name}</p>}
            <p className="text-xs text-[#9A9AA0] font-sans truncate">{user.email}</p>
          </div>
          <button
            onClick={() => { setOpen(false); onLogout(); }}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-blue-400 hover:bg-white/[0.04] font-sans transition-all duration-200"
          >
            <LogOut className="w-3.5 h-3.5" />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}

function AppContent({ user, onLogout }: { user: SessionUser; onLogout: () => void }) {
  const [activePage, setActivePage] = useState('analysis');
  const [activeAgentIds, setActiveAgentIds] = useState<string[]>([]);
  const [agentSummaries, setAgentSummaries] = useState<Record<string, string>>({});

  const [historyRecords, setHistoryRecords] = useState<AnalysisRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API}/analyses`);
      if (res.status === 401) return onLogout();
      const data = await res.json();
      if (data?.success && Array.isArray(data.data)) {
        setHistoryRecords(data.data);
      }
    } catch { /* silent */ } finally {
      setHistoryLoading(false);
    }
  }, [onLogout]);

  // Refresh history when navigating to it, then every 15 seconds while it is open
  useEffect(() => {
    if (activePage !== 'history') return;
    fetchHistory();
    const interval = setInterval(fetchHistory, 15000);
    return () => clearInterval(interval);
  }, [activePage, fetchHistory]);

  const handleDeleteAnalysis = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API}/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (res.ok) {
        setHistoryRecords((prev) => prev.filter((r) => r._id !== id));
      }
    } catch (err) {
      console.error('Failed to delete analysis:', err);
    }
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground flex">
      <Sidebar activePage={activePage} onNavigate={setActivePage} agents={AGENTS} activeAgentId={activeAgentIds} agentSummaries={agentSummaries} />
      <main className="ml-64 flex-1 p-6 min-h-screen">
        <div className="flex items-center justify-end gap-3 mb-4">
          <ThemeToggle />
          <ProfileMenu user={user} onLogout={onLogout} />
        </div>
        {activePage === 'analysis' && (
          <ForensicsSection onActiveAgentsChange={setActiveAgentIds} onAgentSummaryChange={setAgentSummaries} />
        )}
        {activePage === 'trending' && (
          <TrendingPage isFullPage={true} onNavigate={setActivePage} />
        )}
        {activePage === 'history' && (
          <HistoryPage
            records={historyRecords}
            loading={historyLoading}
            onRefresh={fetchHistory}
            onDelete={handleDeleteAnalysis}
          />
        )}
      </main>
    </div>
  );
}

function UnauthenticatedView({ onLoginSuccess }: { onLoginSuccess: (user: SessionUser) => void }) {
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  if (showAuth) {
    return <AuthScreen initialMode={authMode} onBackToLanding={() => setShowAuth(false)} onLoginSuccess={onLoginSuccess} />;
  }

  return (
    <LandingPage
      onNavigateToAuth={(mode) => {
        setAuthMode(mode);
        setShowAuth(true);
      }}
    />
  );
}

export default function Page() {
  // undefined = still checking the session cookie; null = signed out
  const [user, setUser] = useState<SessionUser | null | undefined>(undefined);

  useEffect(() => {
    fetch(`${API}/auth/me`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setUser)
      .catch(() => setUser(null));
  }, []);

  const logout = useCallback(async () => {
    try { await fetch(`${API}/auth/logout`, { method: 'POST' }); } catch { /* signed out locally anyway */ }
    setUser(null);
  }, []);

  if (user === undefined) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <ErrorBoundary>
      {user ? <AppContent user={user} onLogout={logout} /> : <UnauthenticatedView onLoginSuccess={setUser} />}
    </ErrorBoundary>
  );
}
