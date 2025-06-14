"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChartIcon as ChartLineUp } from "lucide-react"
import { Separator } from "@/components/ui/separator"

export default function SignInPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)

  const handleSignIn = (provider: string) => {
    setIsLoading(true)

    // Simulate authentication delay
    setTimeout(() => {
      // In a real app, this would be an actual authentication flow
      console.log(`Signing in with ${provider}`)
      setIsLoading(false)

      // Redirect to prediction page after "authentication"
      router.push("/predict")
    }, 1500)
  }

  return (
    <div className="container flex items-center justify-center min-h-[calc(100vh-4rem)] py-8 px-4">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1 text-center">
          <div className="flex justify-center mb-2">
            <ChartLineUp className="h-10 w-10 text-primary" />
          </div>
          <CardTitle className="text-2xl font-bold">Sign in to RoyFin AI</CardTitle>
          <CardDescription>Choose your preferred sign in method to access stock predictions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button
            variant="outline"
            className="w-full h-12 text-base flex items-center justify-center gap-3"
            disabled={isLoading}
            onClick={() => handleSignIn("Google")}
          >
            <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" className="h-5 w-5">
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                fill="#EA4335"
              />
              <path d="M1 1h22v22H1z" fill="none" />
            </svg>
            Continue with Google
          </Button>

          <Button
            variant="outline"
            className="w-full h-12 text-base flex items-center justify-center gap-3"
            disabled={isLoading}
            onClick={() => handleSignIn("GitHub")}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24">
              <path
                fillRule="evenodd"
                d="M12 2.25c-5.25 0-9 3.75-9 8.25 0 3.75 2.25 6.75 5.25 7.875.375.075.525-.15.525-.375v-1.125c-2.175.375-2.625-1.05-2.625-1.05-.375-.975-.9-1.275-.9-1.275-.675-.45.075-.45.075-.45.825 0 1.275.9 1.275.9.675 1.275 1.725.9 2.175.675.075-.525.3-.9.525-1.125-1.725-.225-3.375-.9-3.375-3.975 0-.9.225-1.725.675-2.475-.075-.15-.3-.825.075-1.725 0 0 .825-.225 2.625.9.825-.225 1.725-.375 2.625-.375.9 0 1.8.15 2.625.375 1.8-1.125 2.625-.9 2.625-.9.375.9.15 1.575.075 1.725.45.75.675 1.575.675 2.475 0 3.075-1.65 3.75-3.375 3.975.375.3.675.825.675 1.575v2.25c0 .225.15.45.525.375C18.75 17.25 21 14.25 21 10.5c0-4.5-3.75-8.25-9-8.25z"
                clipRule="evenodd"
              />
            </svg>
            Continue with GitHub
          </Button>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <Separator className="w-full" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background px-2 text-muted-foreground">Or</span>
            </div>
          </div>

          <Button
            variant="outline"
            className="w-full h-12 text-base flex items-center justify-center gap-3"
            disabled={isLoading}
            onClick={() => handleSignIn("Email")}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24">
              <path
                fillRule="evenodd"
                d="M12 2.25C6.75 2.25 2.25 6.75 2.25 12c0 5.25 4.5 9.75 9.75 9.75 5.25 0 9.75-4.5 9.75-9.75 0-5.25-4.5-9.75-9.75-9.75zm-.375 14.625H9.75v-1.5h1.875v1.5zm0-3.375H9.75V12h1.875v1.5zm0-3.375H9.75V8.25h1.875v1.875zm3.375 6.75h-1.875v-1.5h1.875v1.5zm0-3.375h-1.875V12h1.875v1.875zm0-3.375h-1.875V8.25h1.875v1.875z"
                clipRule="evenodd"
              />
            </svg>
            Continue with Email
          </Button>
        </CardContent>
        <CardFooter className="flex flex-col">
          <p className="text-xs text-center text-muted-foreground mt-2">
            By continuing, you agree to our{" "}
            <Link href="#" className="underline underline-offset-2 hover:text-primary">
              Terms of Service
            </Link>{" "}
            and{" "}
            <Link href="#" className="underline underline-offset-2 hover:text-primary">
              Privacy Policy
            </Link>
            .
          </p>
        </CardFooter>
      </Card>
    </div>
  )
}
