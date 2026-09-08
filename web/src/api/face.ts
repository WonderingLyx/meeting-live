import { call } from './client'

export interface FaceModelInfo {
  type: string
  name: string
  provider: string
  model: string
  embedding_dim?: number
  description?: string
  description_en?: string
  available?: boolean
  dependency_available?: boolean
  install_hint?: string | null
}

export interface FaceDependencyStatus {
  insightface_version?: string | null
  onnxruntime_version?: string | null
  onnxruntime_gpu_version?: string | null
  onnxruntime_directml_version?: string | null
  opencv_version?: string | null
  opencv_error?: string | null
  available_providers: string[]
  dependency_available: boolean
}

export interface FaceSettings {
  provider: string
  endpoint?: string
  api_key_configured?: boolean
  api_key_preview?: string | null
  model: string
  model_id?: string
  device: 'auto' | 'cpu' | 'cuda' | 'directml' | string
  enabled: boolean
  loaded: boolean
  loaded_provider: string
  root: string
  match_threshold: number
  match_margin: number
  frame_interval_sec: number
  last_error?: string | null
  config_path?: string
  config_source?: string
  dependencies: FaceDependencyStatus
  models: Record<string, FaceModelInfo>
}

export async function getFaceSettings() {
  return call<FaceSettings>({ url: '/v1/face/settings' })
}

export async function saveFaceSettings(payload: Partial<FaceSettings> & {
  api_key?: string | null
}) {
  return call<FaceSettings>({
    url: '/v1/face/settings',
    method: 'PUT',
    data: payload,
    timeout: 0,
  })
}

export async function testFaceSettings() {
  return call<FaceSettings>({
    url: '/v1/face/test',
    method: 'POST',
    timeout: 0,
  })
}
