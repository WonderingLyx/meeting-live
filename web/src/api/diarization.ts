import { call } from './client'

export interface DiarizationSettings {
  enabled: boolean
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
  model_revision?: string | null
  last_error?: string | null
  env_path: string
  terms_url: string
  token_url: string
}

export interface DiarizationSettingsPayload {
  provider?: string | null
  endpoint?: string | null
  api_key?: string | null
  hf_token?: string | null
  device: 'auto' | 'cpu' | 'cuda'
  model_id?: string | null
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
