'use client'

import { useState, useEffect } from 'react'
import { useTheme } from 'next-themes'
import { Sun, Moon } from 'lucide-react'

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Avoid rendering theme-dependent UI until mounted -- next-themes
  // doesn't know the real theme during SSR, and guessing wrong causes
  // a flash/hydration mismatch.
  useEffect(() => setMounted(true), [])

  if (!mounted) {
    return <div className="w-9 h-9 rounded-lg bg-white/[0.04] border border-white/[0.08]" />
  }

  const isDark = theme === 'dark'

  return (
    <button
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className="w-9 h-9 rounded-lg bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] hover:border-white/[0.16] flex items-center justify-center transition-all duration-200"
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {isDark ? (
        <Sun className="w-4 h-4 text-[#9A9AA0]" />
      ) : (
        <Moon className="w-4 h-4 text-[#9A9AA0]" />
      )}
    </button>
  )
}
