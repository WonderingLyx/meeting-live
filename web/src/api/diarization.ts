import { call } from './client'

export interface DiarizationSettings {
  enabled: boolean
  engine: string
  engine_info?: DiarizationEngineInfo
  engines?: Record<string, DiarizationEngineInfo>
  token_configured: boolean
  token_preview?: string | null
  token_source?: string | null
  provider?: string
  endpoint?: string
  api_key_configured?: boolean
  api_key_preview?: string | null
  config_path?: string
  config_source?: string
  device: 'auto' | 'cpu' | 'cuda'
  loaded_device: string
  model_id: string
  command?: string
  model_revision?: string | null
  last_error?: string | null
  env_path: string
  terms_url: string
  token_url: string
}

export interface DiarizationEngineInfo {
  type: string
  name: string
  provider?: string
  endpoint?: string
  model?: string
  dependency?: string
  dependency_available?: boolean
  available?: boolean
  install_hint?: string
  requires_token?: boolean
  token_env?: string
  local_after_download?: boolean
  runtime?: string
  languages?: string
  languages_en?: string
  description?: string
  description_en?: string
  recommended_for?: string[]
  experimental?: boolean
  terms_url?: string
}

export interface DiarizationSettingsPayload {
  engine?: string | null
  provider?: string | null
  endpoint?: string | null
  api_key?: string | null
  hf_token?: string | null
  device: 'auto' | 'cpu' | 'cuda'
  model_id?: string | null
  command?: string | null
}

export async function getDiarizationSettings() {
  return call<DiarizationSettings>({ url: '/v1/diarization/settings' })
}

export async function saveDiarizationSettings(data: DiarizationSettingsPayload) {
  return call<DiarizationSettings>({
    url: '/v1/diarization/settings',
    method: 'PUT',
    data,
  })
}

export async function testDiarizationSettings() {
  return call<DiarizationSettings>({
    url: '/v1/diarization/test',
    method: 'POST',
    timeout: 0,
  })
}
