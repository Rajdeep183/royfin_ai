import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { Header } from "@/components/header"
import { TooltipProvider } from "@/components/ui/tooltip"
import { ErrorBoundary } from "@/components/error-boundary"
import { Toaster } from "@/components/ui/sonner"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "RoyFin AI - Intelligent Stock Predictions",
  description: "Advanced AI-powered stock market forecasting and investment intelligence platform with real-time data and ML predictions.",
  keywords: "stock prediction, AI trading, market analysis, investment, machine learning",
  authors: [{ name: "RoyFin AI Team" }],
  openGraph: {
    title: "RoyFin AI - Intelligent Stock Predictions",
    description: "Advanced AI-powered stock market forecasting platform",
    type: "website",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ErrorBoundary>
          <ThemeProvider 
            attribute="class" 
            defaultTheme="system" 
            enableSystem 
            disableTransitionOnChange
          >
            <TooltipProvider>
              <div className="relative flex min-h-screen flex-col">
                <Header />
                <main className="flex-1">
                  <ErrorBoundary>
                    {children}
                  </ErrorBoundary>
                </main>
              </div>
              <Toaster />
            </TooltipProvider>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  )
}