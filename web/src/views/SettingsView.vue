<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { getAsrSettings, getEngines, getModels, saveAsrSettings, switchAsrEngine, switchEngine, testAsrEngine, type AsrInfo, type AsrSettings, type AsrTestResponse, type EngineInfo, type ModelsInfo } from '../api/engines'
import { getLlmSettings, getLlmStatus, saveLlmSettings, testLlmConnection, getLlmPrompts, saveLlmPrompts, listLlmModels, type LlmSettings, type LlmPrompts } from '../api/llm'
import { getDiarizationSettings, saveDiarizationSettings, testDiarizationSettings, type DiarizationSettings } from '../api/diarization'
type LlmResp = Awaited<ReturnType<typeof getLlmStatus>>
type ErrorRecord = {
  id: number
  at: string
  message: string
  method?: string
  url?: string
  status?: number
}
import { useDialog } from '../composables/useDialog'
import EmText from '../components/EmText.vue'
import PromptEditorDialog from '../components/PromptEditorDialog.vue'

const { t, locale } = useI18n()
const dialog = useDialog()

const engines = ref<Record<string, EngineInfo>>({})
const currentEngine = ref<string | null>(null)
const models = ref<ModelsInfo | null>(null)
const llm = ref<LlmResp | null>(null)
const llmSettings = ref<LlmSettings | null>(null)
const llmPrompts = ref<LlmPrompts | null>(null)
const llmModelOptions = ref<Array<{ id: string; label: string }>>([])
const llmModelSource = ref('')
const llmModelError = ref('')
const diarization = ref<DiarizationSettings | null>(null)
const asrSettings = ref<AsrSettings | null>(null)
const hfToken = ref('')
const savingLlm = ref(false)
const savingPrompts = ref(false)
const loadingLlmModels = ref(false)
const savingDiarization = ref(false)
const savingAsrDevice = ref(false)
const testingDiarization = ref(false)
const showPrompts = ref(false)
const showErrorPanel = ref(false)
const testingLlm = ref(false)
const testingAsr = ref(false)
const switchingEngine = ref<string | null>(null)
const switchingAsr = ref<string | null>(null)
const errorRecords = ref<ErrorRecord[]>([])
const asrTestFile = ref<File | null>(null)
const asrTestResult = ref<AsrTestResponse | null>(null)

const engineList = computed(() => {
  const order = ['campplus', 'campplus_cn_en', 'eres2net_large', 'eres2net', 'eres2net_base', 'ecapa_tdnn', 'wespeaker']
  const e = engines.value
  return [...order.filter((k) => e[k]), ...Object.keys(e).filter((k) => !order.includes(k))]
})

const asrEngines = computed<Record<string, AsrInfo>>(() => models.value?.asr_engines?.engines || {})
const currentAsr = computed(() => models.value?.asr_engines?.current || models.value?.asr?.type || 'sensevoice_zh')
const cachedAsr = computed(() => new Set(models.value?.asr_engines?.cached || [currentAsr.value]))
const backendAsrSwitching = computed(() => !!models.value?.asr_engines?.switching)
const pendingAsr = computed(() => switchingAsr.value || models.value?.asr_engines?.pending || null)
const asrBusy = computed(() => !!pendingAsr.value || backendAsrSwitching.value)
const pendingAsrLabel = computed(() => {
  const key = pendingAsr.value
  if (!key) return ''
  return asrEngines.value[key]?.name || key
})
const asrProgress = computed(() => models.value?.asr_engines?.progress || null)
const asrProgressPercent = computed(() => {
  const raw = asrProgress.value?.percent
  if (typeof raw !== 'number' || !Number.isFinite(raw)) return null
  return Math.max(0, Math.min(100, raw))
})
const asrProgressBarStyle = computed(() => (
  asrProgressPercent.value === null ? {} : { width: `${asrProgressPercent.value}%` }
))
const asrDevice = computed(() => asrSettings.value?.device || 'auto')
const asrDeviceStatus = computed(() => asrSettings.value?.device_status || null)
const asrGpuAvailable = computed(() => !!asrDeviceStatus.value?.cuda_available)
const asrGpuLabel = computed(() => {
  const status = asrDeviceStatus.value
  if (!status?.torch_available) return 'PyTorch 未加载'
  if (status.hip_version) return `ROCm ${status.hip_version}`
  if (status.cuda_version) return `CUDA ${status.cuda_version}`
  return asrGpuAvailable.value ? 'GPU 可用' : 'GPU 不可用'
})
const asrGpuNames = computed(() => {
  const devices = asrDeviceStatus.value?.devices || []
  return devices.map((d) => d.name).filter(Boolean).join(' / ')
})
const asrList = computed(() => {
  const order = ['sensevoice_zh', 'paraformer_full', 'paraformer', 'qwen3', 'paraformer_large', 'sensevoice', 'paraformer_spk', 'paraformer_streaming']
  const e = asrEngines.value
  return [...order.filter((k) => e[k]), ...Object.keys(e).filter((k) => !order.includes(k))]
})
const isEnglish = computed(() => String(locale.value).startsWith('en'))

function asrLanguages(info?: AsrInfo) {
  if (!info) return '—'
  return isEnglish.value ? (info.languages_en || info.languages || '—') : (info.languages || info.languages_en || '—')
}

function asrDescription(info?: AsrInfo) {
  if (!info) return ''
  return isEnglish.value ? (info.description_en || info.description || '') : (info.description || info.description_en || '')
}

function asrUnavailableMessage(info?: AsrInfo) {
  if (!info) return ''
  if (isEnglish.value) return info.install_hint_en || info.reason || asrDescription(info)
  return info.install_hint || asrDescription(info)
}

function yesNo(v: unknown) {
  return v ? (t('common.yes') || 'Yes') : (t('common.no') || 'No')
}

function formatBytes(v?: number | null) {
  if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) return ''
  const units = ['B', 'KB', 'MB', 'GB']
  let n = v
  let i = 0
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024
    i += 1
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

function asrDeviceName(device: string) {
  if (device === 'cpu') return 'CPU'
  if (device === 'cuda') return 'GPU'
  if (device === 'mps') return 'MPS'
  return 'Auto'
}

function recordError(detail: Partial<ErrorRecord> | string) {
  const normalized = typeof detail === 'string' ? { message: detail } : detail
  const message = normalized.message || '未知错误'
  errorRecords.value = [
    {
      id: Date.now() + Math.random(),
      at: normalized.at || new Date().toISOString(),
      message,
      method: normalized.method,
      url: normalized.url,
      status: normalized.status,
    },
    ...errorRecords.value,
  ].slice(0, 20)
}

function onApiError(event: Event) {
  recordError((event as CustomEvent<Partial<ErrorRecord>>).detail || '接口请求失败')
}

function clearErrors() {
  errorRecords.value = []
}

function formatErrorTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString(locale.value)
}

function asrCapabilityNote(info?: AsrInfo) {
  const caps = info?.capabilities || {}
  const note = isEnglish.value ? (caps.notes_en || caps.notes) : (caps.notes || caps.notes_en)
  const parts = [
    `${t('settings.asr.cap.upload') || 'Upload'}: ${yesNo(caps.upload)}`,
    `${t('settings.asr.cap.realtime') || 'Realtime'}: ${yesNo(caps.realtime_segmented)}`,
    `${t('settings.asr.cap.words') || 'Word timestamps'}: ${yesNo(caps.word_timestamps)}`,
    `${t('settings.asr.cap.speaker') || 'Speaker ID'}: ${yesNo(caps.speaker_diarization)}`,
  ]
  if (caps.true_streaming === 'adapter_not_yet') {
    parts.push(t('settings.asr.cap.streamingAdapter') || 'Token streaming: adapter not yet')
  } else {
    parts.push(`${t('settings.asr.cap.tokenStreaming') || 'Token streaming'}: ${yesNo(caps.true_streaming)}`)
  }
  if (info?.customized) {
    parts.push(t('settings.asr.cap.customized') || 'Customized')
  }
  if (note) parts.push(String(note))
  return parts.join(' · ')
}

function engineSpeed(info?: EngineInfo) {
  if (!info) return '—'
  return isEnglish.value ? (info.speed_en || info.speed || '—') : (info.speed || info.speed_en || '—')
}

function engineDescription(info?: EngineInfo) {
  if (!info) return ''
  return isEnglish.value ? (info.description_en || info.description || '') : (info.description || info.description_en || '')
}

const llmProviders = [
  { key: 'ollama', label: 'Ollama', endpoint: 'http://127.0.0.1:11434/v1', model: 'qwen2.5:1.5b', allowPublic: false },
  { key: 'lmstudio', label: 'LM Studio', endpoint: 'http://127.0.0.1:1234/v1', model: 'qwen2.5-1.5b-instruct', allowPublic: false },
  { key: 'localai', label: 'LocalAI', endpoint: 'http://127.0.0.1:8080/v1', model: 'qwen2.5:1.5b', allowPublic: false },
  { key: 'vllm', label: 'vLLM', endpoint: 'http://127.0.0.1:8000/v1', model: 'Qwen/Qwen2.5-1.5B-Instruct', allowPublic: false },
  { key: 'openai', label: 'OpenAI-compatible', endpoint: 'https://api.openai.com/v1', model: 'gpt-4o-mini', allowPublic: true },
  { key: 'custom', label: 'Custom', endpoint: '', model: '', allowPublic: false },
]

let asrPollTimer: number | null = null

async function refreshLlmModels(showToast = false) {
  const settings = llmSettings.value
  if (!settings || loadingLlmModels.value) return
  try {
    loadingLlmModels.value = true
    llmModelError.value = ''
    const result = await listLlmModels({
      provider: settings.provider,
      endpoint: settings.endpoint,
      allow_public: settings.allow_public,
    })
    llmModelOptions.value = result.items || []
    llmModelSource.value = result.source || ''
    if (result.error) {
      llmModelError.value = result.error
      if (showToast) window.toast?.(`模型列表获取失败: ${result.error}`, 'error')
    } else if (showToast) {
      window.toast?.(`已获取 ${llmModelOptions.value.length} 个模型`, 'ok')
    }
  } catch (e) {
    llmModelOptions.value = []
    llmModelSource.value = ''
    llmModelError.value = e instanceof Error ? e.message : String(e)
    if (showToast) window.toast?.(`模型列表获取失败: ${llmModelError.value}`, 'error')
  } finally {
    loadingLlmModels.value = false
  }
}

async function refreshModels() {
  try {
    models.value = await getModels()
  } catch {
    models.value = null
  }
}

async function refreshAsrSettings() {
  try {
    asrSettings.value = await getAsrSettings()
  } catch {
    asrSettings.value = null
  }
}

async function load() {
  try {
    const r = await getEngines()
    engines.value = r.engines
    currentEngine.value = r.current
  } catch { /* 后端 8000 不通时静默 */ }
  await refreshModels()
  await refreshAsrSettings()
  try {
    diarization.value = await getDiarizationSettings()
  } catch {
    diarization.value = null
  }
  try {
    llm.value = await getLlmStatus()
    llmSettings.value = await getLlmSettings()
  } catch {
    llm.value = null
    llmSettings.value = null
  }
  try {
    llmPrompts.value = await getLlmPrompts()
  } catch {
    llmPrompts.value = null
  }
}

async function pickEngine(key: string) {
  if (key === currentEngine.value || switchingEngine.value) return
  const info = engines.value[key]
  const ok = await dialog.showConfirm({
    title: t('settings.engine.switchTitle', info?.name || key) || `切换到 ${info?.name || key}?`,
    message: t('settings.engine.switchMessage') || '切换声纹引擎会重新加载对应模型,完成前请避免开始新的识别任务。',
    detail: [
      `${t('settings.engine.params') || '参数量'}: ${info?.params || '—'}`,
      `${t('settings.engine.speed') || '速度'}: ${engineSpeed(info)}`,
      engineDescription(info),
    ].filter(Boolean).join('\n'),
    confirmText: t('settings.engine.switchConfirm') || '切换',
    cancelText: t('btn.cancel') || '取消',
    danger: false,
  })
  if (!ok) return
  try {
    switchingEngine.value = key
    window.toast?.(t('settings.engine.switching', info?.name || key) || `正在切换声纹引擎: ${info?.name || key}`, 'info')
    await switchEngine(key)
    currentEngine.value = key
    models.value = await getModels()
    window.toast?.(t('settings.engine.switched', key) || `已切换到 ${key}`, 'ok')
  } catch (e) {
    window.toast?.(`${t('toast.error')}: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    switchingEngine.value = null
  }
}

async function pickAsr(key: string) {
  if (key === currentAsr.value || asrBusy.value) return
  const info = asrEngines.value[key]
  if (info?.available === false) {
    window.toast?.(info.install_hint || info.reason || (t('settings.asr.unavailable') || '当前环境不可用'), 'error')
    return
  }
  const cached = cachedAsr.value.has(key)
  const ok = await dialog.showConfirm({
    title: t('settings.asr.switchTitle', info?.name || key) || `切换到 ${info?.name || key}?`,
    message: cached
      ? (t('settings.asr.switchCached') || '该模型已加载,切换通常很快。')
      : (t('settings.asr.switchDownload') || '如果本机还没有该模型,后端会先下载并加载。下载/加载期间旧 ASR 会继续工作,完成后才切换。'),
    confirmText: t('settings.asr.switchConfirm') || '切换',
    cancelText: t('btn.cancel') || '取消',
    danger: false,
  })
  if (!ok) return
  try {
    switchingAsr.value = key
    window.toast?.(t('settings.asr.switching', info?.name || key) || `正在切换 ASR: ${info?.name || key}`, 'info')
    await switchAsrEngine(key)
    models.value = await getModels()
    window.toast?.(t('settings.asr.switched', info?.name || key) || `ASR 已切换到 ${info?.name || key}`, 'ok')
  } catch (e) {
    window.toast?.(`${t('settings.asr.switchFail') || 'ASR 切换失败'}: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    switchingAsr.value = null
  }
}

function onAsrTestFile(event: Event) {
  const input = event.target as HTMLInputElement
  asrTestFile.value = input.files?.[0] || null
  asrTestResult.value = null
}

async function runAsrSampleTest() {
  if (!asrTestFile.value || testingAsr.value || asrBusy.value) return
  try {
    testingAsr.value = true
    asrTestResult.value = await testAsrEngine(currentAsr.value, asrTestFile.value)
    window.toast?.(`ASR 测试完成: ${asrTestResult.value.result?.text ? '有识别文本' : '无文本'}`, 'ok')
  } catch (e) {
    window.toast?.(`ASR 测试失败: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    testingAsr.value = false
  }
}

async function pickAsrDevice(device: 'auto' | 'cpu' | 'cuda') {
  if (savingAsrDevice.value || asrBusy.value || device === asrDevice.value) return
  if (device === 'cuda' && !asrGpuAvailable.value) {
    window.toast?.('当前 PyTorch 没有检测到可用 GPU。AMD ROCm 安装成功后这里会显示 ROCm/GPU。', 'error')
    return
  }
  const ok = await dialog.showConfirm({
    title: `切换 ASR 运行设备到 ${asrDeviceName(device)}?`,
    message: `将写入 ASR_DEVICE=${device} 并重新加载当前 ASR 引擎 ${asrEngines.value[currentAsr.value]?.name || currentAsr.value}。`,
    detail: [
      `当前设备: ${asrSettings.value?.loaded_device || asrDevice.value}`,
      `目标后端: ${device === 'cuda' ? asrGpuLabel.value : asrDeviceName(device)}`,
      asrGpuNames.value ? `GPU: ${asrGpuNames.value}` : '',
      '加载失败时会自动回滚到原设备。',
    ].filter(Boolean).join('\n'),
    confirmText: '保存并重载',
    cancelText: t('btn.cancel') || '取消',
    danger: false,
  })
  if (!ok) return
  try {
    savingAsrDevice.value = true
    switchingAsr.value = currentAsr.value
    window.toast?.(`正在重载 ASR 到 ${asrDeviceName(device)}`, 'info')
    asrSettings.value = await saveAsrSettings({ device, reload_current: true })
    models.value = await getModels()
    window.toast?.(`ASR 已切换到 ${asrDeviceName(device)}`, 'ok')
  } catch (e) {
    window.toast?.(`ASR 设备切换失败: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    switchingAsr.value = null
    savingAsrDevice.value = false
  }
}

async function saveDiarizationConfig() {
  if (!diarization.value || savingDiarization.value) return
  try {
    savingDiarization.value = true
    diarization.value = await saveDiarizationSettings({
      hf_token: hfToken.value.trim() || null,
      device: diarization.value.device,
      model_id: diarization.value.model_id,
    })
    hfToken.value = ''
    window.toast?.('说话人分离配置已保存', 'ok')
  } catch (e) {
    window.toast?.(`保存失败: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    savingDiarization.value = false
  }
}

async function testDiarizationConfig() {
  if (!diarization.value || testingDiarization.value) return
  const ok = await dialog.showConfirm({
    title: '加载 pyannote 说话人分离?',
    message: `测试会尝试加载 ${diarization.value.model_id}。首次使用可能联网下载模型；需要先在 Hugging Face 接受模型条款并配置 HF_TOKEN。`,
    detail: `当前 token: ${diarization.value.token_configured ? '已配置' : '未配置'}\n当前设备: ${diarization.value.device}`,
    confirmText: '开始测试',
    cancelText: t('btn.cancel') || '取消',
    danger: false,
  })
  if (!ok) return
  try {
    testingDiarization.value = true
    diarization.value = await testDiarizationSettings()
    window.toast?.(diarization.value.enabled ? 'pyannote 已可用' : (diarization.value.last_error || 'pyannote 不可用'), diarization.value.enabled ? 'ok' : 'error')
  } catch (e) {
    window.toast?.(`测试失败: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    testingDiarization.value = false
  }
}

function isGatedRepoError(message?: string | null) {
  return !!message && /403|gated repo|authorized list|访问被拒绝/i.test(message)
}

function toggleLlm(on: boolean) {
  if (llmSettings.value) {
    llmSettings.value.enabled = on
  }
}

function pickLlmProvider(providerKey: string) {
  if (!llmSettings.value) return
  const preset = llmProviders.find((p) => p.key === providerKey)
  llmSettings.value.provider = providerKey
  if (preset && providerKey !== 'custom') {
    llmSettings.value.endpoint = preset.endpoint
    llmSettings.value.model = preset.model
    llmSettings.value.allow_public = preset.allowPublic
  }
  llmModelOptions.value = []
  llmModelSource.value = ''
  llmModelError.value = ''
}

async function saveLlmConfig() {
  if (!llmSettings.value) return
  try {
    savingLlm.value = true
    await saveLlmSettings(llmSettings.value)
    llm.value = await getLlmStatus()
    llmSettings.value = await getLlmSettings()
    window.toast?.(t('settings.llm.savedPassive') || 'LLM 配置已保存，未发起连接测试', 'ok')
  } catch (e) {
    window.toast?.(`${t('settings.llm.saveFail') || '保存失败'}: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    savingLlm.value = false
  }
}

async function savePrompts(payload: LlmPrompts) {
  try {
    savingPrompts.value = true
    llmPrompts.value = await saveLlmPrompts(payload)
    showPrompts.value = false
    window.toast?.(t('settings.llm.promptsSaved') || 'Prompt 模板已保存', 'ok')
  } catch (e) {
    window.toast?.(`${t('settings.llm.promptsSaveFail') || 'Prompt 保存失败'}: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    savingPrompts.value = false
  }
}

async function testLlmConfig() {
  if (!llm.value?.enabled || testingLlm.value) return
  const ok = await dialog.showConfirm({
    title: t('view.settings.llm.test') || '测试连接',
    message: t('settings.llm.testNotice', llm.value.endpoint || '—')
      || `测试将向已保存的 ${llm.value.endpoint || '—'} 发送一次最小 ping 请求。`,
    detail: t('settings.llm.testPrivacy') || '不会发送会议文稿，但可能产生极少量 API 用量。',
    confirmText: t('settings.llm.testConfirm') || '继续测试',
    cancelText: t('btn.cancel') || '取消',
    danger: false,
  })
  if (!ok) return
  try {
    testingLlm.value = true
    llm.value = await testLlmConnection()
    window.toast?.(
      llm.value.available
        ? (t('settings.llm.testSuccess') || 'LLM 连接可用')
        : (llm.value.error || t('settings.llm.testFailed') || 'LLM 连接不可用'),
      llm.value.available ? 'ok' : 'error',
    )
  } catch (e) {
    window.toast?.(`${t('settings.llm.testFailed') || '连接测试失败'}: ${e instanceof Error ? e.message : e}`, 'error')
  } finally {
    testingLlm.value = false
  }
}

function formatTestedAt(value?: string | null) {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(locale.value)
}

function showLlmHelp() {
  dialog.showConfirm({
    title: t('settings.llm.helpTitle') || '本地 LLM 配置说明',
    message: t('settings.llm.helpBody') || '支持任意 OpenAI-compatible 接口规范',
    detail: [
      'Ollama    →  http://127.0.0.1:11434/v1   (默认端口 11434)',
      '                ollama pull qwen2.5:1.5b',
      'LM Studio →  http://127.0.0.1:1234/v1    (默认端口 1234)',
      'LocalAI   →  http://127.0.0.1:8080/v1',
      'vLLM      →  http://127.0.0.1:8000/v1    (自部署)',
      'OpenRouter / 自建反向代理 (OpenAI 协议)',
      '',
      '在 .env 中设置:',
      '  LLM_ENABLED=true',
      '  LLM_ENDPOINT=http://127.0.0.1:11434/v1',
      '  LLM_MODEL=qwen2.5:1.5b',
      '',
      '重启 Matrix Live Diarizer 即生效。',
    ].join('\n'),
    confirmText: t('btn.confirm') || '知道了',
    cancelText: t('btn.cancel') || '关闭',
    danger: false,
  })
}

onMounted(() => {
  window.addEventListener('app-api-error', onApiError as EventListener)
  load()
  asrPollTimer = window.setInterval(() => {
    if (asrBusy.value) {
      refreshModels()
    }
  }, 2000)
})

onUnmounted(() => {
  window.removeEventListener('app-api-error', onApiError as EventListener)
  if (asrPollTimer !== null) {
    window.clearInterval(asrPollTimer)
    asrPollTimer = null
  }
})
</script>

<template>
  <section class="set-wrap">
    <header class="set-head">
      <h1 class="page-title"><EmText :text="t('view.settings.title')" /></h1>
      <p class="page-sub">{{ t('view.settings.sub') || '声纹引擎 / 本地 LLM / 历史存储' }}</p>
    </header>
    <section class="capability-map" aria-labelledby="capability-title">
      <div class="capability-intro"><span>HOW IT WORKS</span><h2 id="capability-title">{{ t('product.settings.capabilities') }}</h2><p>{{ t('product.settings.capabilitiesHint') }}</p></div>
      <article><b>01</b><h3>{{ t('product.settings.transcription') }}</h3><p>{{ t('product.settings.transcriptionHint') }}</p></article>
      <article><b>02</b><h3>{{ t('product.settings.diarization') }}</h3><p>{{ t('product.settings.diarizationHint') }}</p></article>
      <article><b>03</b><h3>{{ t('product.settings.identity') }}</h3><p>{{ t('product.settings.identityHint') }}</p></article>
    </section>
    <div class="set-grid">

    <!-- ASR 引擎 -->
    <div class="set-row asr-row">
      <div class="l">
        <span>{{ t('settings.asr.label') || 'ASR 引擎' }}</span>
        <em>ASR</em>
      </div>
      <div class="d">{{ t('settings.asr.desc') || '当前语音识别模型由后端 ASR_ENGINE 环境变量决定,修改后需重启服务。' }}</div>
      <div v-if="asrSettings" class="asr-device-panel">
        <div class="asr-device-head">
          <span>运行设备</span>
          <code>ASR_DEVICE={{ asrSettings.device }}</code>
        </div>
        <div class="toggle-group asr-device-toggle" :class="{ busy: savingAsrDevice || asrBusy }">
          <button type="button" :class="{ active: asrDevice === 'auto' }" :disabled="savingAsrDevice || asrBusy" @click="pickAsrDevice('auto')">
            <span class="r" />
            <span>Auto</span>
          </button>
          <button type="button" :class="{ active: asrDevice === 'cpu' }" :disabled="savingAsrDevice || asrBusy" @click="pickAsrDevice('cpu')">
            <span class="r" />
            <span>CPU</span>
          </button>
          <button type="button" :class="{ active: asrDevice === 'cuda' }" :disabled="savingAsrDevice || asrBusy || !asrGpuAvailable" @click="pickAsrDevice('cuda')">
            <span class="r" />
            <span>GPU</span>
          </button>
        </div>
        <div class="asr-device-status">
          <span>{{ asrGpuLabel }}</span>
          <span v-if="asrGpuNames">· {{ asrGpuNames }}</span>
          <span v-if="asrSettings.loaded_device">· 已加载: {{ asrSettings.loaded_device }}</span>
          <span v-if="asrSettings.device_status?.torch_version">· torch {{ asrSettings.device_status.torch_version }}</span>
          <span v-if="asrSettings.device_status?.error">· {{ asrSettings.device_status.error }}</span>
        </div>
      </div>
      <div class="eng-list compact">
        <div
          v-for="key in asrList"
          :key="key"
          class="eng-row"
          :class="{ active: key === currentAsr, switching: key === pendingAsr, disabled: asrEngines[key]?.available === false || (asrBusy && key !== pendingAsr) }"
          @click="pickAsr(key)"
        >
          <div class="radio" />
          <div class="info">
            <div class="n">{{ asrEngines[key]?.name || key }}</div>
            <div class="m">
              <b>{{ asrLanguages(asrEngines[key]) }}</b>
              <span class="sep">·</span>
              {{ asrEngines[key]?.supports_streaming ? (t('settings.asr.streaming') || '实时') : (t('settings.asr.offline') || '离线') }}
              <span v-if="asrEngines[key]?.optional_dependency" class="sep">·</span>
              <span v-if="asrEngines[key]?.optional_dependency">{{ asrEngines[key]?.optional_dependency }}</span>
              <span v-if="asrEngines[key]?.available === false" class="sep">·</span>
              <span v-if="asrEngines[key]?.available === false">{{ t('settings.asr.unavailable') || '不可用' }}</span>
              <span v-if="!cachedAsr.has(key)" class="sep">·</span>
              <span v-if="!cachedAsr.has(key)">{{ t('settings.asr.notCached') || '未加载' }}</span>
            </div>
            <div class="m muted">{{ asrEngines[key]?.available === false ? asrUnavailableMessage(asrEngines[key]) : asrDescription(asrEngines[key]) }}</div>
            <div class="m model-source">{{ t('product.settings.modelId') }}: {{ asrEngines[key]?.model || t('product.settings.modelIdUnavailable') }}</div>
            <div class="m capability">{{ asrCapabilityNote(asrEngines[key]) }}</div>
          </div>
          <span v-if="key === currentAsr" class="pill">{{ t('settings.asr.current') || '当前' }}</span>
          <span v-else-if="key === pendingAsr" class="pill">{{ t('settings.asr.switchingShort') || '切换中' }}</span>
        </div>
        <div v-if="asrList.length === 0" class="empty">
          {{ t('settings.engine.loading') }}
        </div>
      </div>
      <div v-if="asrBusy" class="asr-progress">
        <div class="asr-progress-head">
          <b>ASR 模型正在下载 / 加载</b>
          <span v-if="asrProgressPercent !== null">{{ asrProgressPercent.toFixed(1) }}%</span>
          <span v-else>{{ pendingAsrLabel || '准备中' }}</span>
        </div>
        <div class="asr-progress-sub">
          <span>{{ pendingAsrLabel || asrProgress?.label || 'ASR' }}</span>
          <code v-if="asrProgress?.downloaded_bytes && asrProgress?.total_bytes">
            {{ formatBytes(asrProgress.downloaded_bytes) }} / {{ formatBytes(asrProgress.total_bytes) }}
          </code>
        </div>
        <div
          class="asr-progress-bar"
          :class="{ indeterminate: asrProgressPercent === null }"
          aria-hidden="true"
        >
          <i :style="asrProgressBarStyle" />
        </div>
        <div class="asr-progress-meta">
          当前仍使用 {{ asrEngines[currentAsr]?.name || currentAsr }}，新模型完成后会自动切换。下载期间请保持服务窗口运行。
        </div>
      </div>
      <div class="config-hint">
        <code>ASR_ENGINE={{ currentAsr }}</code>
        <code v-if="pendingAsr">PENDING={{ pendingAsr }}</code>
        <span>{{ t('settings.asr.dynamicHint') || '点击可动态切换;新模型加载完成前继续使用旧 ASR' }}</span>
      </div>
      <div class="asr-test-panel">
        <div class="asr-test-head">
          <b>ASR 样本测试</b>
          <span>{{ currentAsr }}</span>
        </div>
        <div class="asr-test-actions">
          <label class="asr-test-picker">
            <input type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg" @change="onAsrTestFile" />
            <span>{{ asrTestFile?.name || '选择音频样本' }}</span>
          </label>
          <button class="btn ghost sm" type="button" :disabled="!asrTestFile || testingAsr || asrBusy" @click="runAsrSampleTest">
            {{ testingAsr ? '测试中...' : '测试当前 ASR' }}
          </button>
        </div>
        <div v-if="asrTestResult" class="asr-test-result">
          <div>
            <code>{{ asrTestResult.engine_type }}</code>
            <span>{{ asrTestResult.duration_sec.toFixed(1) }}s</span>
          </div>
          <p>{{ asrTestResult.result?.text || '无识别文本' }}</p>
        </div>
      </div>
      <p class="run-provenance-hint">{{ t('product.settings.futureRunsHint') }}</p>
    </div>

    <!-- 声纹引擎 -->
    <div class="set-row engine-row">
      <div class="l">
        <span>{{ t('settings.engine.label') }}</span>
        <em>{{ t('view.settings.engine') }}</em>
      </div>
      <div class="d">{{ t('settings.engine.desc') }}</div>
      <div class="eng-list">
        <div
          v-for="key in engineList"
          :key="key"
          class="eng-row"
          :class="{ active: key === currentEngine, switching: key === switchingEngine }"
          @click="pickEngine(key)"
        >
          <div class="radio" />
          <div class="info">
            <div class="n">{{ engines[key]?.name || key }}</div>
            <div class="m">
              <b>{{ engines[key]?.params || '—' }}</b>
              <span class="sep">·</span>
              {{ engineSpeed(engines[key]) }}
              <span class="sep">·</span>
              {{ engineDescription(engines[key]) }}
            </div>
            <div class="m model-source">{{ t('product.settings.modelId') }}: {{ engines[key]?.model || t('product.settings.modelIdUnavailable') }}</div>
          </div>
          <div
            class="swatch"
            :style="{ background: ['#FF6B35', '#4ECDC4', '#D4A574', '#C589E8', '#7BC96F'][Object.keys(engines).indexOf(key) % 5] }"
          />
          <span v-if="key === currentEngine" class="pill">{{ t('settings.asr.current') || '当前' }}</span>
          <span v-else-if="key === switchingEngine" class="pill">{{ t('settings.asr.switchingShort') || '切换中' }}</span>
        </div>
        <div v-if="engineList.length === 0" class="empty">
          {{ t('settings.engine.loading') }}
        </div>
      </div>
      <p class="run-provenance-hint">{{ t('product.settings.voiceSuggestionHint') }}</p>
    </div>

    <!-- 说话人分离 -->
    <div v-if="diarization" class="set-row diarization-row">
      <div class="l llm-head">
        <span class="llm-title"><span>说话人分离</span><em>PYANNOTE</em></span>
        <span class="diag-badge" :class="{ on: diarization.enabled }">{{ diarization.enabled ? '可用' : '未启用' }}</span>
      </div>
      <div class="d">上传会议的多人分段依赖 pyannote。实时录音的声纹引擎只能做临时聚类和已登记人物匹配，不能替代完整会议的多人分离。</div>
      <div class="llm-form">
        <label>
          <span>Hugging Face Token</span>
          <input v-model.trim="hfToken" type="password" autocomplete="off" placeholder="hf_... 留空则保留已有 token" />
        </label>
        <label>
          <span>pyannote 模型</span>
          <input v-model.trim="diarization.model_id" type="text" autocomplete="off" placeholder="pyannote/speaker-diarization-community-1" />
        </label>
        <label>
          <span>运行设备</span>
          <select v-model="diarization.device">
            <option value="auto">auto</option>
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
          </select>
        </label>
        <div class="llm-secret-note">
          token 会写入本项目 .env。使用前需要先打开模型页面接受条款，再到 Hugging Face 创建 Read token。
        </div>
      </div>
      <div class="diag-links">
        <a :href="diarization.terms_url" target="_blank" rel="noopener">接受模型条款</a>
        <a :href="diarization.token_url" target="_blank" rel="noopener">创建 HF token</a>
        <code>{{ diarization.env_path }}</code>
      </div>
      <div class="llm-options llm-actions">
        <button class="btn primary sm" type="button" :disabled="savingDiarization" @click="saveDiarizationConfig">
          {{ savingDiarization ? '保存中…' : '保存配置' }}
        </button>
        <button class="btn ghost sm" type="button" :disabled="testingDiarization" @click="testDiarizationConfig">
          {{ testingDiarization ? '加载中…' : '测试加载 pyannote' }}
        </button>
      </div>
      <div class="llm-status diag-status">
        <span>Token:</span>
        <b :class="diarization.token_configured ? 'on' : 'off'">{{ diarization.token_configured ? '已配置' : '未配置' }}</b>
        <span v-if="diarization.token_preview" class="model">· {{ diarization.token_preview }}</span>
        <span class="model">· model={{ diarization.model_id }}</span>
        <span class="model">· device={{ diarization.device }} / loaded={{ diarization.loaded_device }}</span>
      </div>
      <div v-if="diarization.last_error" class="llm-error">{{ diarization.last_error }}</div>
      <div v-if="isGatedRepoError(diarization.last_error)" class="llm-secret-note danger">
        403 表示当前 HF_TOKEN 所属账号还没有获得该 gated 模型访问权限。请用同一个 Hugging Face 账号打开模型页面接受/申请访问；如果使用 fine-grained token，还要允许读取公开 gated repositories。
      </div>
    </div>

    <!-- LLM -->
    <div v-if="llm && llmSettings" class="set-row llm-row">
      <div class="l llm-head">
        <span class="llm-title"><span>{{ t('view.settings.llm') }}</span><em>LLM</em></span>
        <button class="btn ghost sm help" type="button" @click="showLlmHelp" :title="t('settings.llm.help') || '配置说明'">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3M12 17h.01" />
          </svg>
          {{ t('settings.llm.help') || '配置说明' }}
        </button>
      </div>
      <div class="d">{{ t('view.settings.llm.desc') }}</div>
      <div class="toggle-group">
        <button
          :class="{ active: !llmSettings.enabled }"
          type="button"
          @click="toggleLlm(false)"
        >
          <span class="r" />
          <span>{{ t('settings.llm.off') }}</span>
        </button>
        <button
          :class="{ active: llmSettings.enabled }"
          type="button"
          @click="toggleLlm(true)"
        >
          <span class="r" />
          <span>{{ t('settings.llm.on') }}</span>
        </button>
      </div>
      <div class="llm-form">
        <label>
          <span>{{ t('settings.llm.provider') || '提供商' }}</span>
          <select v-model="llmSettings.provider" @change="pickLlmProvider(llmSettings.provider)">
            <option v-for="p in llmProviders" :key="p.key" :value="p.key">{{ p.label }}</option>
          </select>
        </label>
        <label>
          <span>Model</span>
          <input v-model.trim="llmSettings.model" list="llm-model-options" placeholder="qwen2.5:1.5b" />
          <datalist id="llm-model-options">
            <option v-for="m in llmModelOptions" :key="m.id" :value="m.id">{{ m.label }}</option>
          </datalist>
        </label>
        <label class="full">
          <span>Endpoint</span>
          <input v-model.trim="llmSettings.endpoint" placeholder="http://127.0.0.1:11434/v1" />
        </label>
        <div class="llm-model-picker">
          <button class="btn ghost sm" type="button" :disabled="loadingLlmModels" @click="refreshLlmModels(true)">
            {{ loadingLlmModels ? '读取模型中' : '刷新可选模型' }}
          </button>
          <span v-if="llmModelOptions.length">{{ llmModelOptions.length }} 个模型 · {{ llmModelSource || 'endpoint' }}</span>
          <span v-else-if="llmModelError">{{ llmModelError }}</span>
          <span v-else>可手动输入，也可从当前 Endpoint 读取</span>
        </div>
        <div class="llm-secret-note">
          {{ llmSettings.has_api_key ? t('settings.llm.keyFromEnv') : t('settings.llm.keyEnvHint') }}
        </div>
      </div>
      <div class="llm-options llm-toggles">
        <label>
          <input v-model="llmSettings.allow_public" type="checkbox" />
          <span>{{ t('settings.llm.allowPublic') || '允许公网 endpoint' }}</span>
        </label>
        <label>
          <input v-model="llmSettings.mock" type="checkbox" />
          <span>{{ t('settings.llm.mock') || 'Mock 模式' }}</span>
        </label>
      </div>
      <div class="llm-options llm-actions">
        <button class="btn primary sm" type="button" :disabled="savingLlm" @click="saveLlmConfig">
          {{ savingLlm ? (t('settings.llm.saving') || '保存中…') : (t('settings.llm.save') || '保存配置') }}
        </button>
        <button
          class="btn ghost sm"
          type="button"
          :disabled="!llm.enabled || savingLlm || testingLlm"
          @click="testLlmConfig"
        >
          {{ testingLlm ? t('view.settings.llm.testing') : t('view.settings.llm.test') }}
        </button>
        <button
          class="btn ghost sm prompts-open"
          type="button"
          :disabled="!llmPrompts"
          @click="showPrompts = true"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="8" y1="13" x2="16" y2="13" />
            <line x1="8" y1="17" x2="13" y2="17" />
          </svg>
          {{ t('settings.llm.promptsOpen') || '编辑 Prompt 模板' }}
        </button>
      </div>
      <div v-if="llm.error" class="llm-error">{{ llm.error }}</div>
      <div class="llm-status">
        <span>{{ t('settings.llm.statusLabel') }}</span>
        <b v-if="llm.available === true" class="on">{{ t('settings.llm.statusAvail') }}</b>
        <b v-else-if="llm.available === false" class="off">{{ t('settings.llm.statusUnavailable') }}</b>
        <b v-else class="off">{{ t('settings.llm.statusNeverTested') }}</b>
        <span v-if="llm.model" class="model">· {{ llm.model }}</span>
        <span class="model">· {{ llm.config_source || 'env' }}</span>
        <span v-if="llm.last_tested_at" class="model">· {{ t('settings.llm.lastTested', formatTestedAt(llm.last_tested_at)) }}</span>
      </div>
    </div>

    <!-- About -->
    <div class="set-row">
      <div class="l"><EmText :text="t('view.settings.about')" /></div>
      <div class="about">
        <span>{{ t('settings.about.text') }}</span><br />
        <a href="/docs" target="_blank" rel="noopener">{{ t('settings.about.docs') }}</a>
        &nbsp;·&nbsp;
        <a href="/health" target="_blank" rel="noopener">Health →</a>
      </div>
    </div>
    </div>
    <div class="error-console">
      <button
        class="error-console-toggle"
        :class="{ has: errorRecords.length > 0, open: showErrorPanel }"
        type="button"
        :aria-expanded="showErrorPanel"
        title="报错窗口"
        @click="showErrorPanel = !showErrorPanel"
      >
        <span>错误</span>
        <b>{{ errorRecords.length }}</b>
      </button>
      <section v-if="showErrorPanel" class="error-console-panel" aria-label="报错窗口">
        <header>
          <strong>最近报错</strong>
          <button type="button" @click="clearErrors">清空</button>
        </header>
        <div v-if="errorRecords.length === 0" class="error-console-empty">
          暂无接口错误
        </div>
        <article v-for="item in errorRecords" :key="item.id">
          <div>
            <time>{{ formatErrorTime(item.at) }}</time>
            <code v-if="item.status">HTTP {{ item.status }}</code>
            <code v-if="item.method || item.url">{{ item.method || 'GET' }} {{ item.url }}</code>
          </div>
          <p>{{ item.message }}</p>
        </article>
      </section>
    </div>
    <PromptEditorDialog
      v-if="showPrompts && llmPrompts"
      :initial="llmPrompts"
      :saving="savingPrompts"
      @close="showPrompts = false"
      @save="savePrompts"
    />
  </section>
</template>

<style scoped>
.capability-map{display:grid;grid-template-columns:1.25fr repeat(3,1fr);gap:1px;margin:0 0 34px;border:1px solid var(--border);background:var(--border);border-radius:10px;overflow:hidden}.capability-map>*{background:var(--ink-2);padding:20px}.capability-intro span,.capability-map article>b{color:var(--amber);font:9px var(--mono);letter-spacing:.12em}.capability-intro h2{font:22px var(--serif);margin:8px 0}.capability-map h3{font-size:13px;margin:10px 0 5px}.capability-map p{color:var(--text-3);font-size:11px;line-height:1.55}@media(max-width:900px){.capability-map{grid-template-columns:1fr 1fr}}@media(max-width:560px){.capability-map{grid-template-columns:1fr}}
.set-wrap { padding: 32px 48px 48px; }
.set-head { padding-bottom: 24px; border-bottom: 1px solid var(--border); margin-bottom: 0; }
/* Page title styles are shared through components.css. */
.set-grid {
  display: grid;
  grid-template-columns: 1.6fr 1fr 1fr;
  grid-auto-rows: auto;
  gap: 16px;
}
.set-grid .set-row { background: var(--ink-2); border: 1px solid var(--border-soft); border-radius: 8px; padding: 22px 24px; }
/* 声纹引擎 (第 1 个 set-row): 占左 1 列, 跨 2 行 (引擎列表高度自适应) */
.set-grid .set-row.engine-row,
.set-grid .set-row.asr-row { grid-row: span 2; }
.set-row { padding: 22px 24px; background: var(--ink-2); }
.set-row + .set-row { border-top: 1px solid var(--border-soft); }
.set-row .l {
  font-family: var(--serif);
  font-size: 18px;
  color: var(--text);
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 4px;
}
.set-row .l span { color: var(--text); }
.set-row .l em { font-style: italic; color: var(--amber); font-variation-settings: 'SOFT' 100, 'WONK' 1; opacity: 0.55; }
.set-row .d {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-3);
  letter-spacing: 0.04em;
  line-height: 1.5;
  margin-bottom: 14px;
}
.eng-list { display: flex; flex-direction: column; gap: 0; border: 1px solid var(--border-soft); border-radius: 8px; overflow: hidden; }
.eng-row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 14px;
  align-items: center;
  padding: 14px 18px;
  background: var(--ink-2);
  cursor: pointer;
  transition: background 0.15s;
  border-bottom: 1px solid var(--border-soft);
}
.eng-row:last-child { border-bottom: none; }
.eng-row:hover { background: var(--ink-3); }
.eng-row.readonly { cursor: default; }
.eng-row.readonly:hover { background: var(--ink-2); }
.eng-row.readonly.active:hover { background: var(--ink-3); }
.eng-row.switching { opacity: 0.72; pointer-events: none; }
.eng-row.disabled { opacity: 0.58; cursor: not-allowed; }
.eng-row.disabled:hover { background: var(--ink-2); }
.eng-row.active { background: var(--ink-3); }
.eng-list.compact .eng-row { padding: 12px 14px; }
.radio { width: 14px; height: 14px; border-radius: 50%; border: 1.5px solid var(--text-3); position: relative; }
.eng-row.active .radio { border-color: var(--amber); }
.eng-row.active .radio::after { content: ''; position: absolute; inset: 3px; border-radius: 50%; background: var(--amber); }
.info .n { font-family: var(--serif); font-size: 16px; color: var(--text); }
.info .m { font-family: var(--mono); font-size: 11px; color: var(--text-2); margin-top: 2px; }
.info .m b { color: var(--teal); font-weight: 500; margin-right: 4px; }
.info .m .sep { color: var(--text-3); margin: 0 4px; }
.info .m.muted { color: var(--text-3); line-height: 1.45; }
.pill {
  padding: 4px 8px;
  border: 1px solid var(--amber);
  border-radius: 999px;
  color: var(--amber);
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.06em;
}

.swatch { width: 4px; height: 28px; border-radius: 2px; }
.toggle-group { display: inline-flex; gap: 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.toggle-group button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: var(--ink-2);
  color: var(--text-2);
  font-family: 'Outfit', sans-serif;
  font-size: 12px;
  font-weight: 500;
  border: none;
  cursor: pointer;
  transition: all 0.15s;
}
.toggle-group button.active { background: var(--amber); color: var(--ink); }
.toggle-group button .r { width: 6px; height: 6px; border-radius: 50%; background: var(--text-3); }
.toggle-group button.active .r { background: var(--ink); }
.toggle-group button:disabled {
  cursor: not-allowed;
  opacity: .48;
}
.asr-device-panel {
  display: flex;
  flex-direction: column;
  gap: 9px;
  margin: 0 0 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-soft);
}
.asr-device-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  font: 10px var(--mono);
  color: var(--text-3);
  letter-spacing: .06em;
}
.asr-device-head code {
  color: var(--amber);
  font: 10px var(--mono);
}
.asr-device-toggle {
  width: fit-content;
}
.asr-device-toggle.busy {
  opacity: .78;
}
.asr-device-status {
  color: var(--text-3);
  font: 10px/1.55 var(--mono);
  overflow-wrap: anywhere;
}
.llm-status {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-2);
  letter-spacing: 0.04em;
}
.llm-status b.on { color: var(--green); }
.llm-status b.off { color: var(--text-3); }
.llm-status .model { color: var(--text-3); }
.diag-badge {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--text-3);
  font: 10px var(--mono);
}
.diag-badge.on { border-color: rgba(78,205,196,.38); color: var(--teal); }
.diag-links {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 10px;
  font: 10px var(--mono);
}
.diag-links a { color: var(--amber); }
.diag-links a:hover { text-decoration: underline; }
.diag-links code {
  max-width: 100%;
  overflow: hidden;
  color: var(--text-3);
  text-overflow: ellipsis;
  white-space: nowrap;
}
.diag-status { flex-wrap: wrap; }
.llm-head { justify-content: space-between; align-items: center; }
.llm-head .llm-title { display: flex; align-items: baseline; gap: 8px; }
.llm-form {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 14px;
}
.llm-form label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.llm-form label.full { grid-column: 1 / -1; }
.llm-form label span,
.llm-options label span {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-3);
  letter-spacing: 0.06em;
}
.model-source{overflow-wrap:anywhere;color:var(--text-3)}
.run-provenance-hint{margin-top:12px;color:var(--text-3);font-size:10px;line-height:1.55}

.llm-secret-note {
  grid-column: 1 / -1;
  color: var(--text-muted);
  font-size: 0.82rem;
  line-height: 1.5;
}
.llm-model-picker {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  min-width: 0;
  color: var(--text-3);
  font: 10px var(--mono);
  overflow-wrap: anywhere;
}
.llm-model-picker span {
  min-width: 0;
  color: var(--text-3);
}
.llm-secret-note.danger {
  margin-top: 8px;
  padding: 9px 10px;
  border: 1px solid rgba(255,95,109,.24);
  border-radius: 6px;
  background: rgba(255,95,109,.08);
  color: #ff8c95;
}
.llm-form input,
.llm-form select {
  width: 100%;
  min-width: 0;
  height: 34px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--ink-1);
  color: var(--text);
  padding: 0 10px;
  font-family: var(--mono);
  font-size: 11px;
}
.llm-options {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
}
.llm-options label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.llm-actions { margin-top: 6px; }
.llm-error {
  margin-top: 10px;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.5;
  color: #ff5f6d;
}
.config-hint {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-3);
  letter-spacing: 0.05em;
}
.config-hint code { color: var(--amber); font-family: var(--mono); font-size: 10px; }
.asr-test-panel {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  background: rgba(255,255,255,.025);
}
.asr-test-head,
.asr-test-actions,
.asr-test-result div {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.asr-test-head b {
  color: var(--text);
  font-size: 12px;
}
.asr-test-head span,
.asr-test-result span {
  color: var(--text-3);
  font: 10px var(--mono);
}
.asr-test-actions {
  flex-wrap: wrap;
  margin-top: 10px;
  justify-content: flex-start;
}
.asr-test-picker {
  position: relative;
  display: inline-flex;
  align-items: center;
  min-width: min(320px, 100%);
  max-width: 100%;
  height: 34px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-2);
  background: var(--ink-1);
  font: 11px var(--mono);
  cursor: pointer;
}
.asr-test-picker input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}
.asr-test-picker span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.asr-test-result {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border-soft);
}
.asr-test-result code {
  color: var(--amber);
  font: 10px var(--mono);
}
.asr-test-result p {
  margin: 8px 0 0;
  color: var(--text-2);
  font-size: 12px;
  line-height: 1.55;
  overflow-wrap: anywhere;
}
.asr-progress {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid rgba(78,205,196,.24);
  border-radius: 6px;
  background: rgba(78,205,196,.07);
}
.asr-progress-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: var(--text);
}
.asr-progress-head b { color: var(--teal); }
.asr-progress-head span {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--amber);
}
.asr-progress-sub {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-top: 8px;
  color: var(--text-2);
  font-family: var(--mono);
  font-size: 10px;
}
.asr-progress-sub code {
  color: var(--teal);
  font-family: var(--mono);
  white-space: nowrap;
}
.asr-progress-bar {
  position: relative;
  height: 7px;
  margin-top: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(255,255,255,.08);
}
.asr-progress-bar i {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  width: 0;
  border-radius: inherit;
  background: linear-gradient(90deg, rgba(78,205,196,0), rgba(78,205,196,.95), rgba(255,107,53,.85));
  transition: width .25s ease;
}
.asr-progress-bar.indeterminate i {
  left: -40%;
  width: 42%;
  animation: asr-progress-scan 1.35s ease-in-out infinite;
}
.asr-progress-meta {
  margin-top: 8px;
  color: var(--text-3);
  font-size: 11px;
  line-height: 1.45;
}
@keyframes asr-progress-scan {
  0% { transform: translateX(0); }
  100% { transform: translateX(335%); }
}
.about { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-2); line-height: 1.7; }
.about a { color: var(--amber); }
.about a:hover { text-decoration: underline; }
.empty { padding: 30px; text-align: center; color: var(--text-3); font-family: var(--mono); font-size: 12px; }
.error-console {
  position: fixed;
  right: 18px;
  bottom: 18px;
  z-index: 60;
  display: flex;
  align-items: flex-end;
  flex-direction: column;
  gap: 8px;
  max-width: min(520px, calc(100vw - 36px));
}
.error-console-toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 34px;
  padding: 0 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--ink-2);
  color: var(--text-3);
  font: 11px var(--mono);
  cursor: pointer;
  box-shadow: 0 8px 24px rgba(0,0,0,.24);
}
.error-console-toggle.has {
  border-color: rgba(255,95,109,.48);
  color: #ff8c95;
}
.error-console-toggle.open {
  color: var(--amber);
}
.error-console-toggle b {
  min-width: 18px;
  height: 18px;
  display: inline-grid;
  place-items: center;
  border-radius: 999px;
  background: rgba(255,255,255,.08);
  color: inherit;
  font-size: 10px;
}
.error-console-panel {
  width: min(520px, calc(100vw - 36px));
  max-height: min(420px, calc(100vh - 100px));
  overflow: auto;
  border: 1px solid rgba(255,95,109,.36);
  border-radius: 8px;
  background: rgba(18,14,12,.98);
  box-shadow: 0 18px 48px rgba(0,0,0,.38);
}
.error-console-panel header {
  position: sticky;
  top: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: rgba(18,14,12,.98);
  border-bottom: 1px solid var(--border-soft);
}
.error-console-panel strong {
  color: var(--text);
  font-size: 12px;
}
.error-console-panel header button {
  border: none;
  background: transparent;
  color: var(--text-3);
  font: 10px var(--mono);
  cursor: pointer;
}
.error-console-empty {
  padding: 18px 12px;
  color: var(--text-3);
  font: 11px var(--mono);
}
.error-console-panel article {
  padding: 11px 12px;
  border-bottom: 1px solid var(--border-soft);
}
.error-console-panel article:last-child {
  border-bottom: none;
}
.error-console-panel article div {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 6px;
}
.error-console-panel time,
.error-console-panel code {
  color: var(--text-3);
  font: 10px var(--mono);
}
.error-console-panel code {
  color: var(--amber);
}
.error-console-panel p {
  margin: 0;
  color: #ff8c95;
  font: 11px/1.5 var(--mono);
  overflow-wrap: anywhere;
}

/* 移动端适配:主网格原本固定三列(1.6fr 1fr 1fr)且无断点,手机上引擎卡片被挤、内容看不全。
   窄屏改单列堆叠,取消 engine/asr-row 的跨行(span 2 在单列无意义且会留空行),
   LLM 表单两列改单列,外边距收紧。eng-row 内部 auto/1fr/auto 不动(外层单列后已有足够宽度)。 */
@media (max-width: 900px) {
  .set-wrap { padding: 24px 20px 32px; }
  .set-grid { grid-template-columns: 1fr; }
  .set-grid .set-row.engine-row,
  .set-grid .set-row.asr-row { grid-row: auto; }
  .llm-form { grid-template-columns: 1fr; }
  .llm-detail .help { margin-left: 0; }
}
</style>
