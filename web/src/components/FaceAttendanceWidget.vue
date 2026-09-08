<script setup lang="ts">
import { computed,nextTick,onBeforeUnmount,onMounted,ref,watch } from 'vue'
import {
  getMeetingAttendance,
  listPeople,
  setManualAttendance,
  submitFaceAttendanceFrame,
  type AttendanceRecord,
  type Person,
} from '../api/product'
import { fmtSec } from '../utils/format'

const props = withDefaults(defineProps<{
  meetingId?: string | null
  timestampSec?: number
  compact?: boolean
  autoIntervalMs?: number
}>(), {
  meetingId: null,
  timestampSec: 0,
  compact: false,
  autoIntervalMs: 5000,
})

const attendance=ref<AttendanceRecord[]>([])
const people=ref<Person[]>([])
const loading=ref(false)
const error=ref('')
const cameraOn=ref(false)
const captureBusy=ref(false)
const video=ref<HTMLVideoElement|null>(null)
const canvas=ref<HTMLCanvasElement|null>(null)
const manualPersonId=ref('')
let stream:MediaStream|null=null
let timer:ReturnType<typeof setInterval>|null=null

const presentIds=computed(()=>new Set(attendance.value.map((item)=>item.person_id)))
const absentPeople=computed(()=>people.value.filter((person)=>!presentIds.value.has(person.id)))
const canUseCamera=computed(()=>!!props.meetingId && !captureBusy.value)
const statusText=computed(()=>{
  if(!props.meetingId)return '等待会议创建'
  if(cameraOn.value)return captureBusy.value?'识别中':'摄像头签到中'
  return '未开启摄像头'
})

function confidenceText(value?:number|null){
  return typeof value==='number'&&Number.isFinite(value)?`${Math.round(value*100)}%`:'手动'
}

async function loadAttendance(){
  if(!props.meetingId)return
  try{
    loading.value=true
    const [attendanceResult,peopleResult]=await Promise.all([
      getMeetingAttendance(props.meetingId),
      listPeople(),
    ])
    attendance.value=attendanceResult.items
    people.value=peopleResult.items
    error.value=''
  }catch(e){
    error.value=e instanceof Error?e.message:String(e)
  }finally{
    loading.value=false
  }
}

function stopCamera(){
  if(timer){clearInterval(timer);timer=null}
  if(stream){stream.getTracks().forEach((track)=>track.stop());stream=null}
  if(video.value)video.value.srcObject=null
  cameraOn.value=false
}

async function startCamera(){
  if(!props.meetingId){error.value='请先开始会议，再启用人脸签到';return}
  if(cameraOn.value)return
  try{
    error.value=''
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:540}},audio:false})
    cameraOn.value=true
    await nextTick()
    if(video.value){
      video.value.srcObject=stream
      await video.value.play()
    }
    await captureFrame()
    timer=setInterval(()=>{void captureFrame()},Math.max(2000,props.autoIntervalMs))
  }catch(e){
    stopCamera()
    error.value=e instanceof Error?e.message:String(e)
  }
}

async function blobFromVideo():Promise<Blob>{
  const v=video.value
  const c=canvas.value
  if(!v||!c||!v.videoWidth||!v.videoHeight)throw new Error('摄像头画面还没有准备好')
  const width=Math.min(960,v.videoWidth)
  const height=Math.round(v.videoHeight*(width/v.videoWidth))
  c.width=width
  c.height=height
  const ctx=c.getContext('2d')
  if(!ctx)throw new Error('无法读取摄像头画面')
  ctx.drawImage(v,0,0,width,height)
  return await new Promise<Blob>((resolve,reject)=>{
    c.toBlob((blob)=>blob?resolve(blob):reject(new Error('截图失败')),'image/jpeg',0.82)
  })
}

async function captureFrame(){
  if(!props.meetingId||captureBusy.value||!cameraOn.value)return
  try{
    captureBusy.value=true
    const blob=await blobFromVideo()
    const result=await submitFaceAttendanceFrame(props.meetingId,blob,props.timestampSec)
    attendance.value=result.attendance
    const names=result.checked_in.map((item)=>item.person_name).filter(Boolean)
    if(names.length)window.toast?.(`人脸签到: ${names.join('、')}`,'ok')
    error.value=''
  }catch(e){
    error.value=e instanceof Error?e.message:String(e)
  }finally{
    captureBusy.value=false
  }
}

async function markManual(){
  if(!props.meetingId||!manualPersonId.value)return
  try{
    const result=await setManualAttendance(props.meetingId,manualPersonId.value,true)
    attendance.value=result.items
    manualPersonId.value=''
    error.value=''
  }catch(e){
    error.value=e instanceof Error?e.message:String(e)
  }
}

watch(()=>props.meetingId,()=>{
  stopCamera()
  attendance.value=[]
  void loadAttendance()
})

onMounted(loadAttendance)
onBeforeUnmount(stopCamera)
</script>

<template>
  <section class="face-attendance" :class="{ compact: props.compact }">
    <header>
      <div>
        <b>人脸签到</b>
        <span>{{ statusText }}</span>
      </div>
      <div class="face-actions">
        <button v-if="!cameraOn" type="button" :disabled="!canUseCamera" @click="startCamera">开启摄像头</button>
        <button v-else type="button" @click="stopCamera">关闭摄像头</button>
        <button type="button" :disabled="!cameraOn||captureBusy" @click="captureFrame">{{ captureBusy?'识别中':'识别一次' }}</button>
      </div>
    </header>
    <div v-if="error" class="face-error">{{ error }}</div>
    <div class="face-body">
      <div class="face-camera">
        <video ref="video" muted playsinline :class="{off:!cameraOn}" />
        <canvas ref="canvas" />
        <span v-if="!cameraOn">{{ props.meetingId?'摄像头未开启':'会议开始后可用' }}</span>
      </div>
      <div class="face-list">
        <div class="face-list-head">
          <strong>已签到 {{ attendance.length }}</strong>
          <small v-if="loading">刷新中</small>
        </div>
        <div v-if="!attendance.length" class="face-empty">暂无签到记录</div>
        <article v-for="item in attendance" :key="item.id">
          <div>
            <b>{{ item.person_name }}</b>
            <span>{{ item.source==='face'?'人脸识别':'手动签到' }} · {{ confidenceText(item.confidence) }}</span>
          </div>
          <time>{{ item.first_seen_sec==null?'—':fmtSec(item.first_seen_sec) }}</time>
        </article>
        <div class="manual-row">
          <select v-model="manualPersonId" :disabled="!props.meetingId">
            <option value="">手动补签</option>
            <option v-for="person in absentPeople" :key="person.id" :value="person.id">{{ person.name }}</option>
          </select>
          <button type="button" :disabled="!manualPersonId" @click="markManual">确认</button>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.face-attendance{margin:18px 0;padding:16px;border:1px solid var(--border-soft);border-radius:8px;background:var(--ink-2)}
.face-attendance header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}
.face-attendance header b{display:block;color:var(--text);font-size:14px}
.face-attendance header span{display:block;margin-top:3px;color:var(--text-3);font:10px var(--mono)}
.face-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.face-actions button,.manual-row button{height:32px;padding:0 10px;border:1px solid var(--border);border-radius:6px;color:var(--amber);font:10px var(--mono)}
.face-actions button:disabled,.manual-row button:disabled{opacity:.45;cursor:not-allowed}
.face-actions button:first-child:not(:disabled),.manual-row button:not(:disabled){background:var(--amber);border-color:var(--amber);color:var(--ink);font-weight:700}
.face-body{display:grid;grid-template-columns:minmax(220px,340px) minmax(0,1fr);gap:14px;align-items:start}
.face-camera{position:relative;aspect-ratio:16/9;overflow:hidden;border:1px solid var(--border);border-radius:8px;background:var(--ink-1)}
.face-camera video{width:100%;height:100%;object-fit:cover;display:block}
.face-camera video.off{opacity:0}
.face-camera canvas{display:none}
.face-camera>span{position:absolute;inset:0;display:grid;place-items:center;color:var(--text-3);font:11px var(--mono)}
.face-list{min-width:0}
.face-list-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
.face-list-head strong{font-size:13px}
.face-list-head small{color:var(--text-3);font:10px var(--mono)}
.face-list article{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid var(--border-soft)}
.face-list article b{display:block;font-size:13px}
.face-list article span{display:block;margin-top:3px;color:var(--text-3);font:10px var(--mono)}
.face-list article time{color:var(--teal);font:10px var(--mono);white-space:nowrap}
.face-empty{padding:18px 0;color:var(--text-3);font:11px var(--mono)}
.manual-row{display:grid;grid-template-columns:minmax(0,1fr) 56px;gap:8px;margin-top:10px}
.manual-row select{height:32px;min-width:0;border:1px solid var(--border);border-radius:6px;background:var(--ink-1);color:var(--text);padding:0 8px;font:11px var(--mono)}
.face-error{margin-bottom:10px;padding:9px 10px;border:1px solid rgba(255,95,109,.3);border-radius:6px;background:rgba(255,95,109,.08);color:#ff8c95;font:11px/1.45 var(--mono);overflow-wrap:anywhere}
.face-attendance.compact{margin:16px 0 22px}
.face-attendance.compact .face-body{grid-template-columns:240px minmax(0,1fr)}
@media(max-width:760px){.face-attendance header{align-items:flex-start;flex-direction:column}.face-body,.face-attendance.compact .face-body{grid-template-columns:1fr}.face-actions{width:100%}.face-actions button{flex:1}}
</style>
