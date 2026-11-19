'use client'

import { useState, useEffect } from 'react'
import ErrorBoundary from '@/components/ErrorBoundary'
import { HydrationGuard } from '@/components/HydrationGuard'
import { ThemeProvider } from '@/components/ThemeProvider'

export default function ClientProviders({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  // During SSR/prerendering, render nothing to avoid context errors
  if (!mounted) return null

  return (
    <ErrorBoundary>
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
        <HydrationGuard>
          {children}
        </HydrationGuard>
      </ThemeProvider>
    </ErrorBoundary>
  )
}
