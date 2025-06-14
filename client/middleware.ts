import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Handle API routes
  if (pathname.startsWith('/api/')) {
    return NextResponse.next()
  }

  // Handle static files
  if (pathname.startsWith('/_next/') || 
      pathname.startsWith('/favicon.ico') ||
      pathname.includes('.')) {
    return NextResponse.next()
  }

  // Ensure proper routing for app pages
  if (pathname === '/') {
    return NextResponse.next()
  }

  // Handle predict page
  if (pathname === '/predict') {
    return NextResponse.next()
  }

  // Handle signin page
  if (pathname === '/signin') {
    return NextResponse.next()
  }

  // For all other routes, let Next.js handle them
  return NextResponse.next()
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}