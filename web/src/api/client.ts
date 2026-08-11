import axios, { AxiosError, type AxiosRequestConfig } from 'axios'
import { useAuthStore } from '../stores/auth'
import { router } from '../router'

export const apiClient = axios.create({
  baseURL: '/',
  timeout: 30_000,
})

function errorMessage(detail: unknown, fallback: string) {
  if (typeof detail === 'string') return detail
  if (detail == null) return fallback
  try {
    return JSON.stringify(detail)
  } catch {
    return fallback
  }
}

function emitApiError(err: AxiosError<{ detail?: unknown }>, config: AxiosRequestConfig, message: string) {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent('app-api-error', {
    detail: {
      at: new Date().toISOString(),
      message,
      method: String(config.method || 'GET').toUpperCase(),
      url: config.url || '',
      status: err.response?.status,
    },
  }))
}

// 请求拦截: Bearer 注入 (除 login;logout 需带 token 才能注销当前会话)
apiClient.interceptors.request.use((config) => {
  const auth = useAuthStore()
  const url = config.url || ''
  if (auth.token && !/\/v1\/auth\/login$/.test(url)) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${auth.token}`
  }
  return config
})

// 响应拦截: 401 → 跳 login
apiClient.interceptors.response.use(
  (r) => r,
  (err: AxiosError) => {
    if (err.response?.status === 401) {
      const auth = useAuthStore()
      auth.clear()
      const next = router.currentRoute.value.fullPath
      router.push({ name: 'login', query: { next } })
    }
    return Promise.reject(err)
  },
)

/** 将后端 detail 统一转换为前端 Error。 */
export async function call<T = unknown>(config: AxiosRequestConfig): Promise<T> {
  try {
    const r = await apiClient.request<T>(config)
    return r.data
  } catch (err) {
    const ax = err as AxiosError<{ detail?: unknown }>
    const detail = errorMessage(ax.response?.data?.detail, ax.message || 'Request failed')
    emitApiError(ax, config, detail)
    throw new Error(detail)
  }
}
