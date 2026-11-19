'use client';
import AppLogo from '@/components/AppLogo';

import React from 'react';
import { Search, Clock, Newspaper, Cpu } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';

interface SidebarProps {
  activePage: string;
  onNavigate: (page: string) => void;
  agents: { id: string; name: string; purpose: string }[];
  activeAgentId: string | string[] | null;
  agentSummaries?: Record<string, string>;
}

const NAV_ITEMS = [
  { key: 'analysis', label: 'Analysis', icon: Search },
  { key: 'trending', label: 'Trending News', icon: Newspaper },
  { key: 'history', label: 'History', icon: Clock },
];

export default function Sidebar({ activePage, onNavigate, agents, activeAgentId, agentSummaries }: SidebarProps) {
  return (
    <div className="w-[260px] h-screen fixed left-0 top-0 flex flex-col z-40 bg-background border-r border-border">
      {/* Logo */}
      <div className="px-6 py-5 flex items-center gap-3 border-b border-border">
        <AppLogo width={36} height={36} />
        <span className="text-xl font-bold tracking-tight text-foreground">VERIF<span className="text-blue-500">AI</span></span>
      </div>

      {/* Navigation */}
      <nav className="px-3 pt-5 pb-2">
        <p className="px-3 mb-2 text-[10px] font-medium uppercase tracking-[0.15em] text-muted-foreground/60">Navigation</p>
        <ul className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = activePage === item.key;
            const Icon = item.icon;
            return (
              <li key={item.key}>
                <button
                  onClick={() => onNavigate(item.key)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 text-[13px] font-medium rounded-xl transition-all duration-300 ${
                    isActive 
                      ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-900/50 shadow-sm' 
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted/50 border border-transparent'
                  }`}
                >
                  <div className={`flex items-center justify-center w-8 h-8 rounded-lg ${isActive ? 'bg-blue-100 dark:bg-blue-900/40' : 'bg-muted/30'}`}>
                    <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-blue-600 dark:text-blue-400' : 'text-muted-foreground/60'}`} />
                  </div>
                  <span className="font-sans">{item.label}</span>
                  {isActive && (
                    <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]" />
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Agents Section */}
      <div className="flex-1 min-h-0 px-3 pt-3 pb-4 flex flex-col border-t border-border mt-2">
        <div className="px-3 mb-3 flex items-center gap-2">
          <Cpu className="w-3 h-3 text-blue-500/60" />
          <span className="text-[10px] font-medium font-sans text-muted-foreground/60 uppercase tracking-[0.15em]">AI Agents</span>
        </div>
        <ScrollArea className="flex-1">
          <div className="space-y-2 px-1">
            {(Array.isArray(agents) ? agents : []).map((agent) => {
              const isRunning = Array.isArray(activeAgentId) ? activeAgentId.includes(agent.id) : activeAgentId === agent.id;
              return (
                <div
                  key={agent.id}
                  className={`px-3 py-2.5 rounded-xl transition-all duration-300 border ${
                    isRunning 
                      ? 'bg-blue-50 dark:bg-blue-900/10 border-blue-200 dark:border-blue-900/30' 
                      : 'bg-muted/20 border-border/50'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <div className="relative flex-shrink-0">
                      <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]' : 'bg-muted-foreground/30'}`} />
                      {isRunning && (
                        <div className="absolute inset-0 w-2 h-2 rounded-full bg-blue-500/40 animate-ping" />
                      )}
                    </div>
                    <span className={`text-xs font-medium font-sans truncate ${isRunning ? 'text-foreground' : 'text-muted-foreground'}`}>{agent.name}</span>
                  </div>
                  <p className="text-[11px] text-muted-foreground/60 mt-1.5 leading-relaxed font-sans pl-[18px]">{agent.purpose}</p>
                  {!isRunning && agentSummaries?.[agent.id] && (
                    <p className="text-[11px] text-[#3B82F6]/80 mt-1 font-sans font-medium pl-[18px]">{agentSummaries[agent.id]}</p>
                  )}
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </div>

      {/* Bottom accent line */}
      <div className="h-[2px] mx-6 mb-4 rounded-full bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />
    </div>
  );
}
