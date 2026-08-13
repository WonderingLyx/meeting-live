import { call } from './client'

export async function getLlmStatus() {
  return call<{
    enabled: boolean
    available: boolean | null
    last_tested_at?: string | null
    endpoint?: string
    model?: string
    mock: boolean
    provider?: string
    allow_public?: boolean
    timeout_sec?: number
    max_input_tokens?: number
    has_api_key?: boolean
    config_source?: string
    config_path?: string
    error?: string | null
    fallback: string
  }>({ url: '/v1/llm/status' })
}

export async function testLlmConnection() {
  return call<Awaited<ReturnType<typeof getLlmStatus>>>({
    url: '/v1/llm/test',
    method: 'POST',
  })
}

export interface LlmSettings {
  provider: string
  enabled: boolean
  endpoint: string
  model: string
  allow_public: boolean
  timeout_sec: number
  max_input_tokens: number
  mock: boolean
  api_key?: string | null
  has_api_key?: boolean
  api_key_preview?: string | null
  config_source?: string
  config_path?: string
}

export interface LlmSettingsPayload extends LlmSettings {
  api_key?: string | null
}

export async function getLlmSettings() {
  return call<LlmSettings>({ url: '/v1/llm/settings' })
}

export async function saveLlmSettings(data: LlmSettingsPayload) {
  return call<LlmSettings>({
    url: '/v1/llm/settings',
    method: 'PUT',
    data,
  })
}

export interface LlmModelList {
  items: Array<{ id: string; label: string }>
  source: string
  error?: string | null
}

export async function listLlmModels(params?: { provider?: string; endpoint?: string; allow_public?: boolean }) {
  return call<LlmModelList>({ url: '/v1/llm/models', params })
}

export interface LlmPrompts {
  summarize: string
  action_items: string
  minutes: string
}

export async function getLlmPrompts() {
  return call<LlmPrompts>({ url: '/v1/llm/prompts' })
}

export async function saveLlmPrompts(data: LlmPrompts) {
  return call<LlmPrompts>({
    url: '/v1/llm/prompts',
    method: 'PUT',
    data,
  })
}
