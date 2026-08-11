import { call } from './client'

export interface EngineInfo {
  name: string
  model: string
  description: string
  description_en?: string
  speed?: string
  speed_en?: string
  params?: string
  eer_voxceleb?: string
  eer_cnceleb?: string
}

export interface AsrInfo {
  name: string
  model: string
  description: string
  description_en?: string
  languages?: string
  languages_en?: string
  supports_streaming?: boolean
  supports_words?: boolean
  optional_dependency?: string
  plugin?: boolean
  plugin_id?: string
  adapter?: string
  config_path?: string
  available?: boolean
  dependency?: string
  reason?: string
  install_hint?: string
  install_hint_en?: string
  python?: string
  type?: string
  word_timestamps_enabled?: boolean
  customized?: boolean
  capabilities?: {
    transcription?: boolean
    upload?: boolean
    realtime_segmented?: boolean
    true_streaming?: boolean | string
    word_timestamps?: boolean
    speaker_diarization?: boolean
    recommended_for?: string[]
    notes?: string
    notes_en?: string
    [key: string]: unknown
  }
}

export interface AsrDownloadProgress {
  engine_type: string
  label?: string
  status?: string
  downloaded_bytes?: number
  total_bytes?: number
  percent?: number | null
}

export interface AsrDeviceStatus {
  torch_available: boolean
  torch_version?: string | null
  backend?: 'cpu' | 'cuda' | 'rocm' | 'mps' | string
  cuda_available: boolean
  cuda_version?: string | null
  hip_version?: string | null
  mps_available?: boolean
  devices?: Array<{ index: number; name: string }>
  error?: string | null
}

export interface AsrSettings {
  device: 'auto' | 'cpu' | 'cuda' | 'mps' | string
  env_device?: string
  loaded_device?: string | null
  env_path?: string
  device_status: AsrDeviceStatus
  reload_result?: AsrSwitchResponse & { reloaded?: boolean }
}

export interface ModelsInfo {
  current: string
  asr: AsrInfo
  asr_engines?: {
    current: string
    engines: Record<string, AsrInfo>
    switching?: boolean
    pending?: string | null
    cached?: string[]
    progress?: AsrDownloadProgress | null
  }
  speakers: Record<string, EngineInfo>
}

export interface AsrSwitchResponse {
  success: boolean
  engine_type: string
  previous_type?: string
  engine_info?: AsrInfo
  already_active?: boolean
  downloaded?: boolean
  switched?: boolean
  error?: string
}

export interface AsrTestResponse {
  engine_type: string
  engine_info: AsrInfo
  sample_rate: number
  duration_sec: number
  result: {
    text?: string
    segments?: unknown[]
    words?: unknown[]
    provider_metadata?: Record<string, unknown>
    [key: string]: unknown
  }
}

export async function getEngines() {
  return call<{ current: string; engines: Record<string, EngineInfo> }>({
    url: '/v1/engines',
  })
}

export async function getModels() {
  return call<ModelsInfo>({
    url: '/v1/models',
  })
}

export async function getAsrEngines() {
  return call<NonNullable<ModelsInfo['asr_engines']>>({
    url: '/v1/asr/engines',
  })
}

export async function getAsrSettings() {
  return call<AsrSettings>({
    url: '/v1/asr/settings',
  })
}

export async function saveAsrSettings(payload: { device: string; reload_current?: boolean }) {
  return call<AsrSettings>({
    url: '/v1/asr/settings',
    method: 'PUT',
    data: payload,
    timeout: 0,
  })
}

export async function switchAsrEngine(engine_type: string) {
  return call<AsrSwitchResponse>({
    url: '/v1/asr/engine',
    method: 'PUT',
    data: { engine_type },
    timeout: 0,
  })
}

export async function testAsrEngine(engine_type: string, file: File) {
  const form = new FormData()
  form.append('engine_type', engine_type)
  form.append('file', file)
  return call<AsrTestResponse>({
    url: '/v1/asr/test',
    method: 'POST',
    data: form,
    timeout: 0,
  })
}

export async function switchEngine(engine_type: string) {
  return call<{ success: boolean; current: string; error?: string }>({
    url: '/v1/engine',
    method: 'PUT',
    data: { engine_type },
  })
}
