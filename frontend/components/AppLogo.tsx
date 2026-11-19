import React from 'react';

export default function AppLogo({ className = "", width = 36, height = 36 }: { className?: string; width?: number; height?: number }) {
  return (
    <svg width={width} height={height} viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      <defs>
        <linearGradient id="shieldGrad" x1="18" y1="2" x2="18" y2="34" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#60A5FA"/>
          <stop offset="100%" stopColor="#2563EB"/>
        </linearGradient>
      </defs>

      <path
        d="M18 2 L32 7 V16 C32 25 26.5 31 18 34 C9.5 31 4 25 4 16 V7 L18 2 Z"
        fill="url(#shieldGrad)"
        fillOpacity="0.12"
        stroke="url(#shieldGrad)"
        strokeWidth="1.6"
      />

      <circle cx="18" cy="16.5" r="6" stroke="#93C5FD" strokeWidth="1.3" fill="none"/>
      <path d="M13.5 21.5 C14.5 19 16 18 18 18 C20 18 21.5 19 22.5 21.5" stroke="#93C5FD" strokeWidth="1.3" fill="none" strokeLinecap="round"/>

      <line x1="7" y1="16.5" x2="29" y2="16.5" stroke="#60A5FA" strokeWidth="1.5" strokeDasharray="1.5 2.2" strokeLinecap="round"/>

      <path d="M9 11 V9 H11" stroke="#DBEAFE" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
      <path d="M27 11 V9 H25" stroke="#DBEAFE" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
      <path d="M9 22 V24 H11" stroke="#DBEAFE" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
      <path d="M27 22 V24 H25" stroke="#DBEAFE" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
    </svg>
  );
}
