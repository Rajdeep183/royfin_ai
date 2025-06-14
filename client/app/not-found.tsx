import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Home, ArrowLeft, Search } from "lucide-react"

export default function NotFound() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-emerald-50 to-slate-50 dark:from-slate-900 dark:via-emerald-900 dark:to-slate-900 flex items-center justify-center px-4">
      <div className="text-center space-y-8 max-w-2xl mx-auto">
        <div className="space-y-4">
          <h1 className="text-8xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent">
            404
          </h1>
          <h2 className="text-3xl font-semibold text-slate-800 dark:text-white">
            Page Not Found
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-400 max-w-md mx-auto">
            The page you're looking for doesn't exist or has been moved. Let's get you back on track.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Button size="lg" className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white px-6 py-3" asChild>
            <Link href="/">
              <Home className="h-4 w-4 mr-2" />
              Go Home
            </Link>
          </Button>
          
          <Button variant="outline" size="lg" className="border-emerald-300 text-emerald-700 hover:bg-emerald-50 dark:border-emerald-300/30 dark:text-emerald-200 dark:hover:bg-emerald-500/10 px-6 py-3" asChild>
            <Link href="/predict">
              <Search className="h-4 w-4 mr-2" />
              Start Predicting
            </Link>
          </Button>
        </div>

        <div className="pt-8">
          <Button variant="ghost" size="sm" asChild>
            <Link href="javascript:history.back()">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Go Back
            </Link>
          </Button>
        </div>
      </div>
    </div>
  )
}