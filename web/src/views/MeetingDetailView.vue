<script setup lang="ts">
import { computed,onBeforeUnmount,onMounted,ref } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { assignSegmentSpeaker,confirmSpeaker,downloadMeeting,generateMeetingNote,getMeeting,getMeetingAudioBlob,listPeople,reprocessMeeting,saveMeetingNote,updateMeeting,updateSegmentText,type MeetingDetail,type Person,type Segment } from '../api/product'
import { fmtSec } from '../utils/format'
import FaceAttendanceWidget from '../components/FaceAttendanceWidget.vue'

const route=useRoute(),{t}=useI18n(),detail=ref<MeetingDetail|null>(null),people=ref<Person[]>([]),loadError=ref('')
const tab=ref<'transcript'|'notes'|'attendance'|'export'>('transcript'),saving=ref(''),audioUrl=ref(''),audioAttempt=ref(0),audioError=ref(''),audioLoading=ref(false),audioReady=ref(false),audioPlaying=ref(false),audioCurrent=ref(0),audioDuration=ref(0)
const selected=ref(new Set<number>()),editingId=ref<number|null>(null),draft=ref('')
const selectedCount=computed(()=>selected.value.size)
const audioDownloadName=computed(()=>`${(detail.value?.meeting.title||'meeting').replace(/[\\/:*?"<>|]+/g,'_')}.wav`)
const noteType=ref<'summary'|'minutes'|'actions'>('summary'),noteDraft=ref(''),noteBusy=ref(false)
const titleEditing=ref(false),titleDraft=ref(''),titleSaving=ref(false)
const reprocessing=ref(false)
const currentNote=computed(()=>detail.value?.notes.find(n=>n.note_type===noteType.value))
const manifest=computed(()=>detail.value?.meeting.processing_manifest)
const transcriptState=computed(()=>detail.value?.meeting.transcript_state||'unknown')
const isRefining=computed(()=>detail.value?.meeting.status==='processing')
const refinementProgress=computed(()=>detail.value?.processing_job?.progress??0)
function refinementStage(){const stage=detail.value?.processing_job?.stage||'queued';const key=`product.job.stage.${stage}`;const translated=t(key);return translated===key?stage:translated}
function noteSource(source:string){if(source==='manual')return t('product.detail.manual');if(source==='llm')return t('product.detail.aiGenerated');if(source==='extractive-fallback')return t('product.detail.localGenerated');return source}
function manifestValue(value?:string){return value||t('product.detail.provenanceUnavailable')}
function diarizationDisplay(){const d=manifest.value?.diarization;if(!d)return undefined;if(d.engine)return d.engine;const model=(d.model||'').toLowerCase();if(model.includes('speech_eres2netv2'))return 'funasr_eres2netv2';if(model.includes('speech_campplus_sv_zh_en'))return 'funasr_campplus_cn_en';if(model.includes('sensevoicesmall'))return 'funasr_sensevoice_campplus';if(model.includes('cam++')||model.includes('speech_campplus'))return 'funasr_campplus';return d.provider}
function alignmentLabel(value?:string){if(!value)return t('product.detail.provenanceUnavailable');const known:Record<string,string>={native:'product.detail.alignmentNative','word-timestamps':'product.detail.alignmentWords','word_timestamp':'product.detail.alignmentWords','word-overlap':'product.detail.alignmentWords','segment-overlap':'product.detail.alignmentSegment','pyannote-segment-overlap':'product.detail.alignmentSegment',none:'product.detail.alignmentNone',anonymous:'product.detail.alignmentNone'};return known[value]?t(known[value]):value}
function qualityLabel(value?:string|null){const known:Record<string,string>={overlap:'疑似多人重叠',noisy:'噪声偏大',clipped:'音频削波',mostly_silent:'有效人声少',uneven_volume:'音量不稳',empty:'空音频'};return value?known[value]||value:''}
function segmentQuality(segment:Segment){if(segment.overlap_flag)return 'overlap';return segment.audio_quality||''}
function segmentQualityScore(segment:Segment){return typeof segment.quality_score==='number'?`${Math.round(segment.quality_score*100)}%`:''}
function audioProcessingDisplay(){const audio=manifest.value?.audio_processing;if(!audio)return undefined;const profile=String(audio.asr_enhancement?.configured_profile||audio.asr_enhancement?.profile||'meeting');const overlap=audio.overlap?.region_count??0;return `${profile} · 疑似重叠 ${overlap} 段`}

let audioRetryTimer:ReturnType<typeof setTimeout>|null=null
let playbackCtx:AudioContext|null=null
let playbackBuffer:AudioBuffer|null=null
let playbackSource:AudioBufferSourceNode|null=null
let playbackStartedAt=0
let playbackOffset=0
let playbackTicker:ReturnType<typeof setInterval>|null=null

function clearAudioUrl(){if(audioUrl.value){URL.revokeObjectURL(audioUrl.value);audioUrl.value=''}}
function cancelAudioRetry(){if(audioRetryTimer){clearTimeout(audioRetryTimer);audioRetryTimer=null}}
function canRetryAudio(){return audioAttempt.value<15&&detail.value?.meeting.status!=='failed'}
function queueAudioRetry(delay?:number){if(audioRetryTimer||!canRetryAudio())return;const attempt=audioAttempt.value+1;audioAttempt.value=attempt;const wait=delay??Math.min(5000,800+attempt*700);audioRetryTimer=setTimeout(()=>{audioRetryTimer=null;void loadAudio()},wait)}
function ensurePlaybackContext(){
  if(playbackCtx)return playbackCtx
  const ctor=window.AudioContext||(window as typeof window&{webkitAudioContext?:typeof AudioContext}).webkitAudioContext
  if(!ctor)throw new Error('当前浏览器不支持 Web Audio 播放')
  playbackCtx=new ctor()
  return playbackCtx
}
function wavChunkId(view:DataView,offset:number){return String.fromCharCode(view.getUint8(offset),view.getUint8(offset+1),view.getUint8(offset+2),view.getUint8(offset+3))}
function wavPcmSample(view:DataView,offset:number,bits:number,format:number){
  if(format===3&&bits===32)return view.getFloat32(offset,true)
  if(format!==1)throw new Error(`不支持的 WAV 编码格式 ${format}`)
  if(bits===8)return (view.getUint8(offset)-128)/128
  if(bits===16)return view.getInt16(offset,true)/32768
  if(bits===24){let value=view.getUint8(offset)|(view.getUint8(offset+1)<<8)|(view.getUint8(offset+2)<<16);if(value&0x800000)value|=0xff000000;return value/8388608}
  if(bits===32)return view.getInt32(offset,true)/2147483648
  throw new Error(`不支持的 WAV 位深 ${bits}`)
}
function decodePcmWav(arrayBuffer:ArrayBuffer,ctx:BaseAudioContext){
  const view=new DataView(arrayBuffer)
  if(view.byteLength<44||wavChunkId(view,0)!=='RIFF'||wavChunkId(view,8)!=='WAVE')throw new Error('会议音频不是标准 WAV 数据')
  let audioFormat=0,channels=0,sampleRate=0,bitsPerSample=0,blockAlign=0,dataOffset=0,dataSize=0
  for(let offset=12;offset+8<=view.byteLength;){
    const id=wavChunkId(view,offset),size=view.getUint32(offset+4,true),start=offset+8
    if(start+size>view.byteLength)break
    if(id==='fmt '){
      audioFormat=view.getUint16(start,true)
      channels=view.getUint16(start+2,true)
      sampleRate=view.getUint32(start+4,true)
      blockAlign=view.getUint16(start+12,true)
      bitsPerSample=view.getUint16(start+14,true)
    }else if(id==='data'){
      dataOffset=start
      dataSize=Math.max(size,view.byteLength-start)
    }
    offset=start+size+(size%2)
  }
  if(!channels||!sampleRate||!bitsPerSample||!dataOffset||!dataSize)throw new Error('会议音频 WAV 头缺少 fmt 或 data 数据')
  const bytesPerSample=Math.ceil(bitsPerSample/8)
  const frameSize=blockAlign||channels*bytesPerSample
  const frames=Math.floor(dataSize/frameSize)
  if(frames<=0)throw new Error('会议音频没有可播放的采样帧')
  const outputChannels=Math.min(channels,2)
  const decoded=ctx.createBuffer(outputChannels,frames,sampleRate)
  const dataEnd=dataOffset+dataSize
  for(let ch=0;ch<outputChannels;ch++){
    const output=decoded.getChannelData(ch)
    for(let frame=0;frame<frames;frame++){
      const sampleOffset=dataOffset+frame*frameSize+ch*bytesPerSample
      output[frame]=sampleOffset+bytesPerSample<=dataEnd?wavPcmSample(view,sampleOffset,bitsPerSample,audioFormat):0
    }
  }
  return decoded
}
async function decodeMeetingAudio(blob:Blob){
  const arrayBuffer=await blob.arrayBuffer()
  const ctx=ensurePlaybackContext()
  try{return await ctx.decodeAudioData(arrayBuffer.slice(0))}catch{return decodePcmWav(arrayBuffer,ctx)}
}
function stopAudioTicker(){if(playbackTicker){clearInterval(playbackTicker);playbackTicker=null}}
function syncAudioClock(){if(!audioPlaying.value||!playbackCtx||!playbackBuffer)return;audioCurrent.value=Math.min(playbackBuffer.duration,playbackOffset+playbackCtx.currentTime-playbackStartedAt)}
function startAudioTicker(){stopAudioTicker();playbackTicker=setInterval(syncAudioClock,120)}
function stopPlaybackSource(){if(playbackSource){playbackSource.onended=null;try{playbackSource.stop()}catch{/* already stopped */}playbackSource.disconnect();playbackSource=null}}
function stopAudio(resetPosition=false){syncAudioClock();stopAudioTicker();audioPlaying.value=false;stopPlaybackSource();if(resetPosition){playbackOffset=0;audioCurrent.value=0}}
function clearDecodedAudio(){stopAudio(true);playbackBuffer=null;audioReady.value=false;audioDuration.value=0}
async function playAudio(){
  if(!playbackBuffer)return
  const ctx=ensurePlaybackContext()
  if(ctx.state==='suspended')await ctx.resume()
  stopPlaybackSource()
  if(playbackOffset>=playbackBuffer.duration)playbackOffset=0
  const source=ctx.createBufferSource()
  source.buffer=playbackBuffer
  source.connect(ctx.destination)
  playbackStartedAt=ctx.currentTime
  playbackSource=source
  audioPlaying.value=true
  source.onended=()=>{if(playbackSource!==source)return;stopAudioTicker();playbackSource=null;audioPlaying.value=false;playbackOffset=0;audioCurrent.value=audioDuration.value;window.setTimeout(()=>{if(!audioPlaying.value)audioCurrent.value=0},150)}
  source.start(0,playbackOffset)
  startAudioTicker()
}
function pauseAudio(){syncAudioClock();playbackOffset=audioCurrent.value;stopAudio(false)}
async function toggleAudio(){try{if(audioPlaying.value)pauseAudio();else await playAudio()}catch(e){audioError.value=e instanceof Error?e.message:String(e)}}
function seekAudio(seconds:number){if(!playbackBuffer)return;const next=Math.max(0,Math.min(seconds,playbackBuffer.duration));const wasPlaying=audioPlaying.value;stopAudio(false);playbackOffset=next;audioCurrent.value=next;if(wasPlaying)void playAudio()}
function seekFromRange(event:Event){seekAudio(Number((event.target as HTMLInputElement).value))}

async function load(){try{detail.value=await getMeeting(String(route.params.id));people.value=(await listPeople()).items;loadError.value=''}catch(e){loadError.value=e instanceof Error?e.message:String(e)}}
async function loadAudio(){cancelAudioRetry();try{audioLoading.value=true;clearDecodedAudio();const blob=await getMeetingAudioBlob(String(route.params.id));const buffer=await decodeMeetingAudio(blob);clearAudioUrl();audioUrl.value=URL.createObjectURL(blob);playbackBuffer=buffer;audioDuration.value=buffer.duration;audioCurrent.value=0;playbackOffset=0;audioReady.value=true;audioAttempt.value=0;audioError.value=''}catch(e){clearDecodedAudio();clearAudioUrl();audioError.value=e instanceof Error?e.message:String(e);queueAudioRetry()}finally{audioLoading.value=false}}
async function assignIdentity(speakerId:string,personId:string){if(isRefining.value)return;saving.value=speakerId;try{await confirmSpeaker(String(route.params.id),speakerId,personId||null);await load()}finally{saving.value=''}}
async function confirmSuggested(speakerId:string,personId?:string){if(personId)await assignIdentity(speakerId,personId)}
function segmentSpeaker(segment:Segment){return ['auto_matched','confirmed'].includes(segment.identity_status||'')&&segment.person_name?segment.person_name:(segment.speaker_label||t('product.detail.unknown'))}
async function reprocess(){reprocessing.value=true;try{await reprocessMeeting(String(route.params.id));await load()}catch(e){window.toast?.(`${t('product.detail.reprocessFailed')}: ${e instanceof Error?e.message:e}`,'error')}finally{reprocessing.value=false}}
function seek(segment:Segment){seekAudio(segment.start_time);void playAudio()}
function beginEdit(segment:Segment){if(isRefining.value)return;editingId.value=segment.id;draft.value=segment.text}
function cancelEdit(){editingId.value=null;draft.value=''}
async function saveEdit(segment:Segment){const text=draft.value.trim();if(!text)return;await updateSegmentText(String(route.params.id),segment.id,text);segment.text=text;segment.manually_edited=1;cancelEdit()}
function toggle(id:number){if(isRefining.value)return;const next=new Set(selected.value);next.has(id)?next.delete(id):next.add(id);selected.value=next}
async function batchAssign(speakerId:string){if(!selected.value.size||!speakerId)return;await assignSegmentSpeaker(String(route.params.id),[...selected.value],speakerId==='__clear__'?null:speakerId);selected.value=new Set();await load()}
function openNote(type:'summary'|'minutes'|'actions'){noteType.value=type;noteDraft.value=detail.value?.notes.find(n=>n.note_type===type)?.content||''}
async function generateNote(){noteBusy.value=true;try{const note=await generateMeetingNote(String(route.params.id),noteType.value);noteDraft.value=note.content;await load()}finally{noteBusy.value=false}}
async function saveNote(){if(!noteDraft.value.trim())return;noteBusy.value=true;try{await saveMeetingNote(String(route.params.id),noteType.value,noteDraft.value.trim());await load()}finally{noteBusy.value=false}}
function beginTitleEdit(){if(!detail.value)return;titleDraft.value=detail.value.meeting.title;titleEditing.value=true}
async function saveTitle(){const title=titleDraft.value.trim();if(!title||!detail.value)return;titleSaving.value=true;try{detail.value.meeting=await updateMeeting(detail.value.meeting.id,title);titleEditing.value=false}finally{titleSaving.value=false}}

let refreshTimer:ReturnType<typeof setInterval>|null=null
onMounted(async()=>{await Promise.all([load(),loadAudio()]);refreshTimer=setInterval(async()=>{const st=detail.value?.meeting.status;if(st!=='processing')return;await load();if(detail.value?.meeting.status==='ready'&&!audioUrl.value)void loadAudio()},2000);const segment=Number(route.query.segment);if(segment){requestAnimationFrame(()=>document.getElementById(`segment-${segment}`)?.scrollIntoView({behavior:'smooth',block:'center'}))}})
onBeforeUnmount(()=>{if(refreshTimer)clearInterval(refreshTimer);cancelAudioRetry();clearDecodedAudio();clearAudioUrl();if(playbackCtx)void playbackCtx.close()})
</script>

<template><section v-if="detail" class="product-page detail-page">
  <header class="detail-head"><div><div class="eyebrow">{{t(detail.meeting.source==='live'?'product.detail.liveRecord':'product.detail.uploadRecord')}}</div><div v-if="titleEditing" class="title-editor"><input v-model="titleDraft" maxlength="200" autofocus @keyup.enter="saveTitle" @keyup.esc="titleEditing=false"><button :disabled="titleSaving" @click="saveTitle">{{t('product.detail.saveTitle')}}</button><button @click="titleEditing=false">{{t('product.detail.cancel')}}</button></div><button v-else class="title-button" :title="t('product.detail.rename')" @click="beginTitleEdit"><h1>{{detail.meeting.title}}</h1><span>✎</span></button><p>{{t('product.detail.meta',{segments:detail.segments.length,speakers:detail.speakers.length})}} <span class="transcript-state" :class="transcriptState">{{t(`product.detail.state.${transcriptState}`)}}</span></p></div><div class="tabs" role="tablist"><button :class="{active:tab==='transcript'}" role="tab" :aria-selected="tab==='transcript'" @click="tab='transcript'">{{t('product.detail.transcript')}}</button><button :class="{active:tab==='notes'}" role="tab" :aria-selected="tab==='notes'" @click="tab='notes';openNote(noteType)">{{t('product.detail.outputs')}}</button><button :class="{active:tab==='attendance'}" role="tab" :aria-selected="tab==='attendance'" @click="tab='attendance'">人脸签到</button><button :class="{active:tab==='export'}" role="tab" :aria-selected="tab==='export'" @click="tab='export'">{{t('product.detail.export')}}</button></div></header>
  <div v-if="audioReady" class="custom-player" role="group" aria-label="会议录音播放器">
    <button type="button" class="play-toggle" :disabled="audioLoading" :aria-label="audioPlaying?'暂停会议录音':'播放会议录音'" @click="toggleAudio">{{audioPlaying?'暂停':'播放'}}</button>
    <time class="play-time">{{fmtSec(audioCurrent)}} / {{fmtSec(audioDuration)}}</time>
    <input class="playback-range" type="range" min="0" :max="audioDuration||0" step="0.01" :value="audioCurrent" aria-label="会议录音进度" @input="seekFromRange">
    <a v-if="audioUrl" class="download-audio" :href="audioUrl" :download="audioDownloadName">下载音频</a>
  </div>
  <aside v-else-if="audioLoading" class="audio-warning audio-status">
    <strong>正在准备会议音频</strong>
    <p>正在读取录音并转换为页面播放器可用的音频缓冲。</p>
    <small v-if="audioAttempt">已自动重试 {{audioAttempt}} 次</small>
  </aside>
  <aside v-else-if="audioError" class="audio-warning">
    <strong>会议音频暂时无法播放</strong>
    <p>{{audioError}}</p>
    <button type="button" :disabled="audioLoading" @click="loadAudio">{{audioLoading?'重新获取中':'重新获取音频'}}</button>
    <small v-if="audioAttempt">已自动重试 {{audioAttempt}} 次</small>
  </aside>
  <aside v-if="isRefining" class="refining"><strong>{{t('product.detail.refiningTitle')}}</strong><p>{{t('product.detail.refiningHint')}}</p><div class="refinement-progress"><i :style="{width:`${refinementProgress}%`}"/></div><small>{{t('product.detail.refiningProgress',{stage:refinementStage(),progress:refinementProgress})}}</small><p class="draft-lock">{{t('product.detail.draftLocked')}}</p></aside>
  <aside v-else-if="detail.meeting.status==='failed'" class="capability-warning"><div><strong>{{t('product.detail.refinementFailedTitle')}}</strong><p>{{t('product.detail.refinementFailedHint')}}</p></div><code v-if="detail.meeting.error_message">{{detail.meeting.error_message}}</code><button :disabled="reprocessing" @click="reprocess">{{reprocessing?t('product.detail.reprocessing'):t('product.detail.reprocess')}}</button></aside>
  <aside v-else-if="detail.meeting.diarization_status==='unavailable'" class="capability-warning"><div><strong>{{t('product.detail.diarizationTitle')}}</strong><p>{{t('product.detail.diarizationUnavailable')}}</p></div><details><summary>{{t('product.detail.diarizationHow')}}</summary><p>{{t('product.detail.diarizationSetup')}}</p><code>{{detail.meeting.diarization_error||'HF_TOKEN'}}</code></details><button :disabled="reprocessing" @click="reprocess">{{reprocessing?t('product.detail.reprocessing'):t('product.detail.reprocess')}}</button></aside>
  <details class="provenance"><summary>{{t('product.detail.provenanceTitle')}}</summary><p>{{t('product.detail.provenanceHint')}}</p><dl><template v-if="manifest"><dt>{{t('product.detail.provenanceStrategy')}}</dt><dd>{{manifestValue(manifest.strategy)}}</dd><dt>{{t('product.detail.provenanceAsr')}}</dt><dd>{{manifestValue(manifest.asr?.model||manifest.asr?.engine)}}</dd><dt>{{t('product.detail.provenanceTiming')}}</dt><dd>{{manifestValue(manifest.asr?.timestamp_granularity)}}</dd><dt>音频处理</dt><dd>{{manifestValue(audioProcessingDisplay())}}</dd><dt>{{t('product.detail.provenanceDiarization')}}</dt><dd>{{manifestValue(diarizationDisplay())}}</dd><dt>{{t('product.detail.provenanceAlignment')}}</dt><dd>{{alignmentLabel(manifest.diarization?.alignment)}}</dd><dt>{{t('product.detail.provenanceIdentity')}}</dt><dd>{{manifestValue(manifest.speaker_identity?.model_id||manifest.speaker_identity?.engine)}}</dd></template><template v-else><dt>{{t('product.detail.provenanceState')}}</dt><dd>{{t('product.detail.provenanceLegacy')}}</dd></template></dl></details>
  <template v-if="tab==='transcript'">
    <aside v-if="detail.speakers.length" class="speaker-confirm"><h2>{{t('product.detail.confirmSpeakers')}}</h2><p>{{t('product.detail.confirmHint')}}</p><p class="anonymous-hint">{{t('product.detail.anonymousHint')}}</p><label v-for="s in detail.speakers" :key="s.id"><span>{{s.label}}</span><select :value="s.person_id||''" :disabled="isRefining||saving===s.id" @change="assignIdentity(s.id,($event.target as HTMLSelectElement).value)"><option value="">{{t('product.detail.unconfirmed')}}</option><option v-for="p in people" :key="p.id" :value="p.id">{{p.name}}</option></select><em v-if="s.identity_status==='confirmed'">{{t('product.detail.identityConfirmed')}}</em><em v-else-if="s.identity_status==='auto_matched'">{{t('product.detail.identityAutoMatched',{score:Math.round((s.confidence||0)*100)})}}</em><template v-else-if="s.identity_status==='suggested'&&s.person_id"><em>{{t('product.detail.identitySuggested',{score:Math.round((s.confidence||0)*100)})}}</em><button :disabled="isRefining||saving===s.id" @click="confirmSuggested(s.id,s.person_id)">{{t('product.detail.confirmSuggestion')}}</button></template></label></aside>
    <div v-if="selectedCount&&!isRefining" class="batch"><b>{{t('product.detail.selected',{count:selectedCount})}}</b><select @change="batchAssign(($event.target as HTMLSelectElement).value)"><option value="">{{t('product.detail.assignTo')}}</option><option v-for="s in detail.speakers" :key="s.id" :value="s.id">{{s.person_name||s.label}}</option><option value="__clear__">{{t('product.detail.clearSpeaker')}}</option></select><button @click="selected=new Set()">{{t('product.detail.cancelSelection')}}</button></div>
    <main class="transcript"><article v-for="s in detail.segments" :id="`segment-${s.id}`" :key="s.id" :class="{selected:selected.has(s.id),manual:s.manually_edited,overlap:s.overlap_flag}"><input type="checkbox" :disabled="isRefining" :checked="selected.has(s.id)" @change="toggle(s.id)"><button class="who" @click="seek(s)"><b>{{segmentSpeaker(s)}}</b><span v-if="s.identity_status==='auto_matched'" class="identity-badge">{{t('product.detail.identityAutoBadge')}}</span><time>{{fmtSec(s.start_time)}}</time></button><div v-if="editingId===s.id" class="editor"><textarea v-model="draft" rows="3" @keydown.ctrl.enter="saveEdit(s)"/><div><button @click="cancelEdit">{{t('product.detail.cancel')}}</button><button class="save" @click="saveEdit(s)">{{t('product.detail.saveEdit')}}</button></div></div><p v-else @dblclick="beginEdit(s)">{{s.text}}<span v-if="segmentQuality(s)" class="quality-badge" :class="segmentQuality(s)" :title="segmentQualityScore(s)">{{qualityLabel(segmentQuality(s))}}</span><span v-if="s.manually_edited" class="corrected">{{t('product.detail.corrected')}}</span><button v-if="!isRefining" class="edit" @click="beginEdit(s)">{{t('product.detail.edit')}}</button></p></article><div v-if="!detail.segments.length" class="empty-state">{{t(detail.meeting.status==='processing'?'product.detail.processing':'product.detail.noTranscript')}}</div></main>
  </template>
  <section v-else-if="tab==='notes'" class="notes-panel"><div class="note-head"><div><h2>{{t('product.detail.outputsTitle')}}</h2><p>{{t('product.detail.outputsHint')}}</p></div><div class="note-types"><button v-for="type in (['summary','actions','minutes'] as const)" :key="type" :class="{active:noteType===type}" @click="openNote(type)">{{t(`product.detail.${type}`)}}</button></div></div><div v-if="!currentNote&&!noteDraft.trim()" class="note-empty"><span>✦</span><h3>{{t(`product.detail.${noteType}Empty`)}}</h3><p>{{isRefining?t('product.detail.draftLocked'):t('product.detail.generateHint')}}</p><button :disabled="isRefining||noteBusy||!detail.segments.length" @click="generateNote">{{noteBusy?t('product.detail.generating'):t('product.detail.generateNow')}}</button></div><template v-else><textarea v-model="noteDraft" rows="18" :disabled="isRefining" :placeholder="t('product.detail.notePlaceholder')"/><footer><span v-if="currentNote">{{t('product.detail.source',{source:noteSource(currentNote.source)})}}</span><button :disabled="isRefining||noteBusy" @click="generateNote">{{t(currentNote?'product.detail.regenerate':'product.detail.generate')}}</button><button class="save" :disabled="isRefining||noteBusy||!noteDraft.trim()" @click="saveNote">{{t('product.detail.saveNote')}}</button></footer></template></section>
  <section v-else-if="tab==='attendance'" class="attendance-panel">
    <FaceAttendanceWidget :meeting-id="detail.meeting.id" :timestamp-sec="audioCurrent" />
  </section>
  <div v-else class="export-panel"><h2>{{t('product.detail.exportTitle')}}</h2><p>{{t('product.detail.exportHint')}}</p><div><button @click="downloadMeeting(detail.meeting.id,'markdown')"><b>Markdown</b><span>{{t('product.detail.markdownHint')}}</span></button><button @click="downloadMeeting(detail.meeting.id,'srt')"><b>SRT</b><span>{{t('product.detail.srtHint')}}</span></button><button @click="downloadMeeting(detail.meeting.id,'vtt')"><b>WebVTT</b><span>{{t('product.detail.vttHint')}}</span></button><button @click="downloadMeeting(detail.meeting.id,'json')"><b>JSON</b><span>{{t('product.detail.jsonHint')}}</span></button></div></div>
</section><section v-else class="product-page"><div v-if="loadError" class="empty-state detail-error"><b>{{t('product.common.loadFailed')}}</b><span>{{loadError}}</span><button @click="load">{{t('product.common.retry')}}</button></div><div v-else class="empty-state">{{t('product.detail.loading')}}</div></section></template>

<style scoped>
.title-button{display:flex;align-items:center;gap:12px;text-align:left}.title-button span{opacity:0;color:var(--amber)}.title-button:hover span,.title-button:focus span{opacity:1}.title-editor{display:flex;align-items:center;gap:8px;margin:8px 0}.title-editor input{min-width:min(520px,60vw);padding:9px 11px;background:var(--ink-2);border:1px solid var(--amber);border-radius:6px;font:24px var(--serif)}.title-editor button{color:var(--amber);font:10px var(--mono)}.detail-error{display:flex;flex-direction:column;gap:8px}.detail-error b{color:var(--red)}.detail-error button{color:var(--amber)}
.custom-player{display:grid;grid-template-columns:72px 96px minmax(160px,1fr) 86px;align-items:center;gap:12px;margin:18px 0;padding:10px 12px;border:1px solid var(--border);border-radius:8px;background:var(--ink-2)}.play-toggle{height:38px;border-radius:6px;background:var(--amber);color:var(--ink);font:12px var(--mono);font-weight:700}.play-toggle:disabled{opacity:.5}.play-time{font:11px var(--mono);color:var(--text-2);white-space:nowrap}.playback-range{width:100%;accent-color:var(--amber)}.download-audio{display:inline-flex;align-items:center;justify-content:center;height:34px;border:1px solid var(--border);border-radius:6px;color:var(--amber);font:10px var(--mono);white-space:nowrap}.download-audio:hover{border-color:var(--amber);background:var(--amber-soft)}
.detail-head{display:flex;justify-content:space-between;align-items:end;border-bottom:1px solid var(--border);padding-bottom:24px}.detail-head h1{font:42px var(--serif);margin:8px 0}.detail-head p,.speaker-confirm p,.export-panel>p{color:var(--text-3)}.tabs{display:flex;gap:20px}.tabs button{padding:9px 0;border-bottom:2px solid transparent}.tabs button.active{color:var(--amber);border-color:var(--amber)}.audio-warning{display:flex;align-items:center;gap:14px;margin:18px 0;padding:12px 14px;border:1px solid rgba(255,107,53,.42);border-radius:8px;background:var(--amber-soft);color:var(--text-2)}.audio-warning strong{color:var(--amber);white-space:nowrap}.audio-warning p{flex:1;margin:0;color:var(--text-2);font-size:12px;line-height:1.45;overflow-wrap:anywhere}.audio-warning button{padding:7px 10px;border-radius:5px;background:var(--amber);color:var(--ink);font:10px var(--mono);white-space:nowrap}.audio-warning button:disabled{opacity:.55}.audio-warning small{color:var(--text-3);font:10px var(--mono);white-space:nowrap}.speaker-confirm{margin:8px 0 20px;padding:20px;background:var(--ink-2);border-radius:10px}.speaker-confirm h2,.export-panel h2{font:22px var(--serif)}.speaker-confirm label{display:inline-flex;align-items:center;gap:8px;margin:16px 16px 0 0}.speaker-confirm select,.batch select{background:var(--ink-3);color:var(--text);border:1px solid var(--border);padding:7px 10px;border-radius:5px}.batch{position:sticky;top:8px;z-index:3;display:flex;align-items:center;gap:14px;padding:12px 18px;background:var(--gold);color:var(--ink);border-radius:8px}.transcript{max-width:900px;margin:18px auto}.transcript article{display:grid;grid-template-columns:22px 140px 1fr;gap:18px;padding:18px 10px;border-bottom:1px solid var(--border-soft);border-radius:6px}.transcript article.selected{background:var(--amber-soft)}.transcript article.manual{border-left:2px solid var(--teal)}.transcript article.overlap{border-left:2px solid rgba(255,107,53,.85)}.who{display:flex;flex-direction:column;text-align:left}.who b{color:var(--amber)}time{font:10px var(--mono);color:var(--text-3)}.transcript p{font-size:16px;line-height:1.75}.quality-badge{display:inline-flex;align-items:center;margin-left:9px;padding:2px 6px;border-radius:999px;background:var(--ink-3);color:var(--text-3);font:9px var(--mono);vertical-align:middle}.quality-badge.overlap{background:rgba(255,107,53,.14);color:var(--amber)}.quality-badge.noisy,.quality-badge.uneven_volume,.quality-badge.clipped{background:rgba(255,193,7,.12);color:var(--gold)}.quality-badge.mostly_silent,.quality-badge.empty{background:rgba(255,80,105,.12);color:var(--red)}.edit{opacity:0;color:var(--amber);font:10px var(--mono);margin-left:10px}.transcript article:hover .edit{opacity:1}.corrected{color:var(--teal);font:9px var(--mono);margin-left:10px}.editor textarea{width:100%;resize:vertical;background:var(--ink-2);border:1px solid var(--amber);border-radius:6px;padding:10px;line-height:1.6}.editor>div{display:flex;justify-content:flex-end;gap:12px;margin-top:8px}.editor .save{color:var(--amber)}.export-panel{max-width:750px;margin:60px auto}.export-panel>div{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:24px}.export-panel button{padding:20px;background:var(--ink-2);border:1px solid var(--border);border-radius:8px;text-align:left;display:flex;flex-direction:column}.export-panel button:hover{border-color:var(--amber)}.export-panel span{color:var(--text-3);margin-top:4px}@media(max-width:700px){.detail-head{align-items:start;flex-direction:column;gap:20px}.custom-player{grid-template-columns:72px 1fr}.play-time{justify-self:end}.playback-range,.download-audio{grid-column:1/-1}.download-audio{width:100%}.transcript article{grid-template-columns:20px 1fr}.transcript article>p,.editor{grid-column:2}.batch{flex-wrap:wrap}.audio-warning{align-items:flex-start;flex-direction:column}}
.notes-panel,.attendance-panel{max-width:850px;margin:36px auto}.note-types{display:flex;gap:8px;margin-bottom:12px}.note-types button{padding:8px 13px;border-radius:5px;background:var(--ink-2)}.note-types button.active{background:var(--amber-soft);color:var(--amber)}.notes-panel textarea{width:100%;padding:18px;background:var(--ink-2);border:1px solid var(--border);border-radius:8px;line-height:1.7;resize:vertical}.notes-panel footer{display:flex;justify-content:flex-end;align-items:center;gap:14px;margin-top:12px}.notes-panel footer span{margin-right:auto;color:var(--text-3);font:10px var(--mono)}.notes-panel footer button{color:var(--amber)}.notes-panel footer .save{padding:8px 14px;background:var(--amber);color:var(--ink);border-radius:5px}
.capability-warning{max-width:900px;margin:12px auto;padding:14px 16px;border:1px solid rgba(255,107,53,.45);border-radius:8px;background:var(--amber-soft);color:var(--text-2)}.capability-warning strong{color:var(--amber)}.capability-warning p{margin-top:4px}.capability-warning details{margin-top:9px;padding-top:9px;border-top:1px solid rgba(255,107,53,.2)}.capability-warning summary{cursor:pointer;color:var(--text);font:11px var(--mono)}.capability-warning code{display:block;margin-top:7px;color:var(--text-3);white-space:normal}.note-head{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:18px}.note-head h2{font:28px var(--serif)}.note-head p{color:var(--text-3);margin-top:5px}.note-head .note-types{margin-bottom:0}.note-empty{min-height:330px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;border:1px dashed var(--border);border-radius:12px;background:var(--ink-2)}.note-empty>span{color:var(--amber);font-size:28px}.note-empty h3{font:25px var(--serif);margin:12px 0 6px}.note-empty p{max-width:520px;color:var(--text-3)}.note-empty button{margin-top:20px;padding:10px 18px;border-radius:7px;background:var(--amber);color:var(--ink);font-weight:700}.note-empty button:disabled{opacity:.4}@media(max-width:700px){.note-head{align-items:flex-start;flex-direction:column}.title-editor{align-items:flex-start;flex-wrap:wrap}.title-editor input{min-width:100%}}
.capability-warning>button{margin-top:12px;padding:8px 13px;border-radius:6px;background:var(--amber);color:var(--ink);font-weight:700}.refining{max-width:900px;margin:12px auto;padding:14px 16px;border-left:2px solid var(--teal);background:var(--teal-soft)}.refining strong{color:var(--teal)}.refining p{margin-top:4px;color:var(--text-2)}.refining small{display:block;margin-top:6px;color:var(--teal);font:10px var(--mono)}.refining .draft-lock{color:var(--text-3);font-size:11px}.refinement-progress{height:4px;margin-top:12px;overflow:hidden;border-radius:999px;background:var(--ink-3)}.refinement-progress i{display:block;height:100%;background:var(--teal);transition:width .25s}.speaker-confirm label em{color:var(--text-3);font:9px var(--mono);font-style:normal}.speaker-confirm label button{padding:6px 9px;border-radius:5px;background:var(--amber);color:var(--ink);font:9px var(--mono)}.identity-badge{width:max-content;padding:2px 5px;border-radius:999px;background:var(--teal-soft);color:var(--teal);font:8px var(--mono)}
.transcript-state{display:inline-block;margin-left:8px;padding:2px 7px;border:1px solid var(--border);border-radius:999px;font:9px var(--mono);color:var(--text-2)}.transcript-state.refined{border-color:rgba(78,205,196,.45);color:var(--teal)}.provenance{max-width:900px;margin:12px auto;padding:12px 16px;border:1px solid var(--border-soft);border-radius:8px;background:var(--ink-2)}.provenance summary{cursor:pointer;color:var(--text-2);font:11px var(--mono)}.provenance>p{margin:8px 0;color:var(--text-3);font-size:11px}.provenance dl{display:grid;grid-template-columns:minmax(100px,160px) 1fr;gap:7px 16px;margin-top:12px}.provenance dt{color:var(--text-3);font:10px var(--mono)}.provenance dd{overflow-wrap:anywhere;color:var(--text-2);font-size:12px}.anonymous-hint{margin-top:6px;color:var(--amber)!important;font-size:11px}
</style>
