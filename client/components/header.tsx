"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Brain } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full bg-white/95 dark:bg-slate-900/95 backdrop-blur supports-[backdrop-filter]:bg-white/80 dark:supports-[backdrop-filter]:bg-slate-900/80 border-b border-slate-200 dark:border-slate-700/50">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-lg flex items-center justify-center">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <Link href="/" className="text-xl font-bold bg-gradient-to-r from-slate-800 to-emerald-700 dark:from-white dark:to-emerald-200 bg-clip-text text-transparent">
            RoyFin AI
          </Link>
        </div>

        <div className="flex items-center gap-4">
          <Button variant="ghost" className="text-slate-600 hover:text-slate-800 hover:bg-slate-100 dark:text-slate-300 dark:hover:text-white dark:hover:bg-slate-800" asChild>
            <Link href="/predict">
              Dashboard
            </Link>
          </Button>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
