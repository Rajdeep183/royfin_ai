import { useState, useCallback } from 'react'
import { toast } from 'sonner'

export interface ApiError {
  message: string
  code: string
  statusCode?: number
}

export interface UseApiState<T> {
  data: T | null
  loading: boolean
  error: ApiError | null
}

export function useApi<T>() {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null
  })

  const execute = useCallback(async (apiCall: () => Promise<T>, showErrorToast = true) => {
    setState(prev => ({ ...prev, loading: true, error: null }))

    try {
      const data = await apiCall()
      setState({ data, loading: false, error: null })
      return data
    } catch (error: any) {
      const apiError: ApiError = {
        message: error?.message || 'An unexpected error occurred',
        code: error?.code || 'UNKNOWN_ERROR',
        statusCode: error?.statusCode || 500
      }

      setState(prev => ({ ...prev, loading: false, error: apiError }))

      if (showErrorToast) {
        toast.error(apiError.message, {
          description: `Error code: ${apiError.code}`
        })
      }

      throw apiError
    }
  }, [])

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null })
  }, [])

  return {
    ...state,
    execute,
    reset
  }
}