import Link from "next/link"
import { Button } from "@/components/ui/button"
import { TrendingUp, Brain, Target, Zap, BarChart3, ArrowRight } from "lucide-react"

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-emerald-50 to-slate-50 dark:from-slate-900 dark:via-emerald-900 dark:to-slate-900">
      {/* Hero Section */}
      <section className="relative py-20 px-4">
        <div className="absolute inset-0 bg-white/20 dark:bg-black/20"></div>
        <div className="relative max-w-7xl mx-auto">
          <div className="text-center space-y-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-100/80 dark:bg-emerald-500/20 backdrop-blur-sm rounded-full border border-emerald-200 dark:border-emerald-300/30">
              <Brain className="h-4 w-4 text-emerald-600 dark:text-emerald-300" />
              <span className="text-emerald-700 dark:text-emerald-200 text-sm font-medium">AI-Powered Market Intelligence</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-slate-800 via-emerald-700 to-teal-700 dark:from-white dark:via-emerald-200 dark:to-teal-200 bg-clip-text text-transparent leading-tight">
              Predict Market
              <br />
              Movements with AI
            </h1>
            
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto leading-relaxed">
              Harness the power of advanced machine learning algorithms to forecast stock prices, 
              analyze market trends, and make data-driven investment decisions with confidence.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button size="lg" className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white px-8 py-4 text-lg font-semibold rounded-xl shadow-2xl" asChild>
                <Link href="/predict">
                  <Target className="h-5 w-5 mr-2" />
                  Start Predicting
                  <ArrowRight className="h-5 w-5 ml-2" />
                </Link>
              </Button>
              
              <Button variant="outline" size="lg" className="border-emerald-300 text-emerald-700 hover:bg-emerald-50 dark:border-emerald-300/30 dark:text-emerald-200 dark:hover:bg-emerald-500/10 px-8 py-4 text-lg rounded-xl backdrop-blur-sm">
                View Demo
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-800 dark:text-white mb-4">
              Advanced Market Analytics
            </h2>
            <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              Our cutting-edge platform provides institutional-grade tools for retail investors
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="group relative bg-white/70 dark:bg-slate-800/50 backdrop-blur-sm border border-slate-200 dark:border-slate-700/50 rounded-2xl p-8 hover:bg-white dark:hover:bg-slate-800/70 transition-all duration-300 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-100/50 to-teal-100/50 dark:from-emerald-600/10 dark:to-teal-600/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center mb-6">
                  <TrendingUp className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-800 dark:text-white mb-4">
                  Neural Network Predictions
                </h3>
                <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                  Advanced LSTM neural networks analyze historical patterns to predict future price movements with remarkable accuracy.
                </p>
              </div>
            </div>

            <div className="group relative bg-white/70 dark:bg-slate-800/50 backdrop-blur-sm border border-slate-200 dark:border-slate-700/50 rounded-2xl p-8 hover:bg-white dark:hover:bg-slate-800/70 transition-all duration-300 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-100/50 to-teal-100/50 dark:from-emerald-600/10 dark:to-teal-600/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center mb-6">
                  <BarChart3 className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-800 dark:text-white mb-4">
                  Real-time Market Analysis
                </h3>
                <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                  Track earnings announcements, volatility patterns, and market sentiment to stay ahead of major price movements.
                </p>
              </div>
            </div>

            <div className="group relative bg-white/70 dark:bg-slate-800/50 backdrop-blur-sm border border-slate-200 dark:border-slate-700/50 rounded-2xl p-8 hover:bg-white dark:hover:bg-slate-800/70 transition-all duration-300 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-100/50 to-teal-100/50 dark:from-emerald-600/10 dark:to-teal-600/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center mb-6">
                  <Zap className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-800 dark:text-white mb-4">
                  Risk-Adjusted Forecasting
                </h3>
                <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
                  Customize prediction models to match your risk tolerance and investment strategy for optimal decision-making.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-700/50 py-8 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
              <Brain className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              <span className="text-lg font-semibold text-slate-800 dark:text-white">RoyFin AI</span>
            </div>
            <div className="flex gap-6 text-sm text-slate-600 dark:text-slate-400">
              <Link href="#" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">
                Privacy Policy
              </Link>
              <Link href="#" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">
                Terms of Service
              </Link>
              <Link href="#" className="hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors">
                Contact Us
              </Link>
            </div>
          </div>
          <div className="text-center mt-6 pt-6 border-t border-slate-200 dark:border-slate-700/50">
            <p className="text-sm text-slate-500">
              Made with ❤️ by Rajdeep Roy
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
