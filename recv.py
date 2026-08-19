import asyncio
import hashlib
import itertools
import json
import queue
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="SANA Control")
ROOT = Path("sana_received")
ROOT.mkdir(exist_ok=True)
PASSWORD = "wangran"
KEY = hashlib.sha256(PASSWORD.encode()).digest()
clients: set[WebSocket] = set()
history: list[dict] = []
task_queue: queue.Queue = queue.Queue(maxsize=1000)
task_ids = itertools.count(1)


class TaskIn(BaseModel):
    prompt: str
    seed: int | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 2


class BatchIn(BaseModel):
    prompts: list[str]
    seed: int | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 2


def auth(authorization: str | None):
    if authorization != f"Bearer {PASSWORD}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def make_task(prompt: str, width: int, height: int, steps: int, seed: int | None = None):
    return {
        "id": next(task_ids),
        "prompt": prompt,
        "seed": seed if seed is not None else int(time.time_ns() % 2_147_483_647),
        "width": width,
        "height": height,
        "steps": steps,
    }


def enqueue_nowait(task: dict):
    try:
        task_queue.put_nowait(task)
    except queue.Full:
        raise HTTPException(status_code=503, detail="Task queue is full")


def decrypt(data: bytes) -> bytes:
    return AESGCM(KEY).decrypt(data[:12], data[12:], None)


async def broadcast(item: dict):
    dead = []
    payload = json.dumps(item, ensure_ascii=False)
    for ws in list(clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


@app.post("/task", status_code=202)
def add_task(x: TaskIn, authorization: str | None = Header(None)):
    """单个 Prompt：即使包含换行，整个字符串也只创建一个任务。"""
    auth(authorization)
    prompt = x.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")
    task = make_task(prompt, x.width, x.height, x.steps, x.seed)
    enqueue_nowait(task)
    return {"queued": 1, "queue_size": task_queue.qsize(), "task": task}


@app.post("/task/batch", status_code=202)
def add_batch(x: BatchIn, authorization: str | None = Header(None)):
    """批量 Prompt：列表中的每一个元素创建一个任务。"""
    auth(authorization)
    prompts = [p.strip() for p in x.prompts if p.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts")

    items = []
    for i, prompt in enumerate(prompts):
        seed = x.seed + i if x.seed is not None else None
        task = make_task(prompt, x.width, x.height, x.steps, seed)
        enqueue_nowait(task)
        items.append(task)

    return {"queued": len(items), "queue_size": task_queue.qsize(), "tasks": items}


@app.get("/task/next")
async def next_task(authorization: str | None = Header(None)):
    auth(authorization)
    try:
        return await asyncio.to_thread(task_queue.get, True, 25)
    except queue.Empty:
        return Response(status_code=204)


@app.get("/api/status")
def status():
    return {"queued": task_queue.qsize(), "images": len(history), "clients": len(clients)}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    id: int = Form(...),
    gpu: int = Form(...),
    seed: int = Form(...),
    prompt: str = Form(""),
    seconds: float = Form(0),
    authorization: str | None = Header(None),
):
    auth(authorization)
    try:
        encrypted = await file.read()
        data = await asyncio.to_thread(decrypt, encrypted)
    except Exception:
        raise HTTPException(status_code=400, detail="Decrypt failed")

    name = f"{int(time.time()*1000)}_{id:04d}_gpu{gpu}_seed{seed}.webp"
    await asyncio.to_thread((ROOT / name).write_bytes, data)

    item = {
        "id": id,
        "gpu": gpu,
        "seed": seed,
        "prompt": prompt,
        "seconds": seconds,
        "url": f"/images/{name}",
        "time": time.time(),
    }
    history.append(item)
    del history[:-300]
    await broadcast(item)
    return item


@app.get("/api/history")
def get_history():
    return history[-200:]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        clients.discard(websocket)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SANA Control</title>
<style>
*{box-sizing:border-box}
:root{--bg:#11111b;--panel:#181825;--panel2:#1e1e2e;--border:#313244;--text:#cdd6f4;--muted:#7f849c;--blue:#89b4fa;--green:#a6e3a1;--red:#f38ba8;--purple:#cba6f7}
body{margin:0;background:var(--bg);color:var(--text);font:14px Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.shell{max-width:1680px;margin:auto;padding:20px}
.top{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:18px}
.title{font-size:23px;font-weight:800}.sub{color:var(--muted);font-size:12px;margin-top:4px}
.status{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.pill{background:var(--panel2);border:1px solid var(--border);border-radius:999px;padding:7px 11px;color:#bac2de}
.dot{width:8px;height:8px;border-radius:50%;background:var(--red);display:inline-block;margin-right:6px}.online .dot{background:var(--green)}
.layout{display:grid;grid-template-columns:440px minmax(0,1fr);gap:18px;align-items:start}.left{display:flex;flex-direction:column;gap:14px}
.panel,.params{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:16px;box-shadow:0 15px 40px #0003}
.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px}.panel h2,.params h2{margin:0;font-size:15px}
.badge{font-size:10px;font-weight:800;letter-spacing:.06em;color:var(--blue);border:1px solid #89b4fa55;background:#89b4fa12;padding:5px 7px;border-radius:7px;white-space:nowrap}
.batch-badge{color:var(--purple);border-color:#cba6f755;background:#cba6f712}.help{color:var(--muted);font-size:11px;line-height:1.5;margin-bottom:9px}
.label{font-size:10px;color:#9399b2;margin:10px 0 6px;font-weight:700;letter-spacing:.05em}
textarea,input{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px;outline:none}
textarea{resize:vertical;line-height:1.55}textarea:focus,input:focus{border-color:var(--blue)}#singlePrompt{min-height:220px}#batchPrompts{min-height:190px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:9px}.actions{display:flex;gap:8px;margin-top:12px}.btn{border:0;border-radius:10px;padding:10px 13px;font-weight:750;cursor:pointer}
.primary{background:var(--blue);color:var(--bg);flex:1}.batch-primary{background:var(--purple);color:var(--bg);flex:1}.ghost{background:var(--border);color:var(--text)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}.stat{background:var(--bg);border-radius:10px;padding:10px}.stat b{display:block;font-size:17px}.stat span{color:var(--muted);font-size:10px}
.gallery-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.gallery{columns:4 240px;column-gap:12px}
.card{break-inside:avoid;margin-bottom:12px;background:var(--panel2);border:1px solid var(--border);border-radius:13px;overflow:hidden;animation:in .2s ease}.card img{width:100%;display:block}
.info{padding:10px}.meta{font-size:10px;color:var(--blue)}.prompt{white-space:pre-wrap;font-size:12px;color:#bac2de;margin-top:5px;line-height:1.4}.empty{color:#585b70;text-align:center;padding:80px 20px}
.toast{position:fixed;right:20px;bottom:20px;background:var(--green);color:var(--bg);padding:11px 15px;border-radius:10px;font-weight:750;display:none;z-index:10;max-width:420px}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}@media(max-width:980px){.layout{grid-template-columns:1fr}.gallery{columns:2 180px}}
</style>
</head>
<body>
<div class="shell">
  <div class="top">
    <div><div class="title">SANA Control</div><div class="sub">Local Task Queue · Kaggle Dual GPU</div></div>
    <div class="status"><span class="pill" id="conn"><span class="dot"></span>Connecting</span><span class="pill">Queue <b id="q">0</b></span></div>
  </div>

  <div class="layout">
    <aside class="left">

      <section class="panel">
        <div class="panel-head">
          <div><h2>单个 Prompt</h2><div class="sub">允许任意换行与分段</div></div>
          <span class="badge">MULTILINE → 1 TASK</span>
        </div>
        <div class="help">这里无论写多少行，整个文本框都会作为 <b>一个 Prompt / 一个任务</b> 提交。</div>
        <textarea id="singlePrompt" placeholder="例如：

A cinematic portrait of a traveler,
standing in a rainy Tokyo alley,

neon reflections on wet pavement,
35mm photography,
soft volumetric light..."></textarea>
        <div class="actions">
          <button class="btn primary" id="singleSubmit" onclick="submitSingle()">提交 1 个任务</button>
          <button class="btn ghost" onclick="clearSingle()">清空</button>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div><h2>批量 Prompt</h2><div class="sub">每一行独立生成</div></div>
          <span class="badge batch-badge">1 LINE → 1 TASK</span>
        </div>
        <div class="help">只有这个区域会按换行拆分。<b>一行 = 一个 Prompt = 一个任务。</b></div>
        <textarea id="batchPrompts" placeholder="A cinematic mountain lake at sunrise
A futuristic Tokyo street at night
A peaceful forest covered in mist"></textarea>
        <div class="actions">
          <button class="btn batch-primary" id="batchSubmit" onclick="submitBatch()">批量加入队列</button>
          <button class="btn ghost" onclick="clearBatch()">清空</button>
        </div>
      </section>

      <section class="params">
        <div class="panel-head"><div><h2>生成参数</h2><div class="sub">单个 / 批量共用</div></div></div>
        <div class="label">ACCESS TOKEN</div><input id="token" type="password" placeholder="wangran">
        <div class="row">
          <div><div class="label">WIDTH</div><input id="width" type="number" value="1024"></div>
          <div><div class="label">HEIGHT</div><input id="height" type="number" value="1024"></div>
        </div>
        <div class="row">
          <div><div class="label">STEPS</div><input id="steps" type="number" value="2"></div>
          <div><div class="label">BASE SEED · 可空</div><input id="seed" type="number" placeholder="随机"></div>
        </div>
        <div class="stats">
          <div class="stat"><b id="batchCount">0</b><span>BATCH PROMPTS</span></div>
          <div class="stat"><b id="queueCount">0</b><span>QUEUED</span></div>
          <div class="stat"><b id="imageCount">0</b><span>IMAGES</span></div>
        </div>
      </section>
    </aside>

    <section>
      <div class="gallery-head"><div><b>Live Results</b><div class="sub">生成完成后自动出现</div></div><button class="btn ghost" onclick="location.reload()">刷新</button></div>
      <main id="grid" class="gallery"><div class="empty" id="empty">等待第一张图片...</div></main>
    </section>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const $=id=>document.getElementById(id);
const grid=$("grid"), token=$("token"), singlePrompt=$("singlePrompt"), batchPrompts=$("batchPrompts");
token.value=localStorage.getItem("sana_token")||"";
token.addEventListener("change",()=>localStorage.setItem("sana_token",token.value));

function toast(text,bad=false){const el=$("toast");el.textContent=text;el.style.background=bad?"#f38ba8":"#a6e3a1";el.style.display="block";setTimeout(()=>el.style.display="none",2400)}
function commonParams(){const body={width:Number($("width").value||1024),height:Number($("height").value||1024),steps:Number($("steps").value||2)};const seed=$("seed").value.trim();if(seed!=="")body.seed=Number(seed);return body}
function batchLines(){return batchPrompts.value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean)}
function updateBatchCount(){$("batchCount").textContent=batchLines().length}
batchPrompts.addEventListener("input",updateBatchCount);

// 单个 Prompt：这里绝不按换行拆分
async function submitSingle(){
  const prompt=singlePrompt.value.trim();
  if(!prompt)return toast("请输入单个 Prompt",true);
  const body={...commonParams(),prompt};
  const btn=$("singleSubmit");btn.disabled=true;btn.textContent="正在入队...";
  try{
    const r=await fetch("/task",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+token.value},body:JSON.stringify(body)});
    if(!r.ok)throw new Error(await r.text());
    const x=await r.json();
    toast(`单个任务 #${x.task.id} 已入队 · 当前队列 ${x.queue_size}`);
    singlePrompt.value="";
    status();
  }catch(e){toast("单个任务提交失败："+e.message,true)}finally{btn.disabled=false;btn.textContent="提交 1 个任务"}
}

// 批量 Prompt：只有这里按“一行一个”拆分
async function submitBatch(){
  const prompts=batchLines();
  if(!prompts.length)return toast("请输入批量 Prompt",true);
  const body={...commonParams(),prompts};
  const btn=$("batchSubmit");btn.disabled=true;btn.textContent="正在入队...";
  try{
    const r=await fetch("/task/batch",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+token.value},body:JSON.stringify(body)});
    if(!r.ok)throw new Error(await r.text());
    const x=await r.json();
    toast(`已批量加入 ${x.queued} 个任务 · 当前队列 ${x.queue_size}`);
    batchPrompts.value="";updateBatchCount();status();
  }catch(e){toast("批量提交失败："+e.message,true)}finally{btn.disabled=false;btn.textContent="批量加入队列"}
}

function clearSingle(){singlePrompt.value="";singlePrompt.focus()}
function clearBatch(){batchPrompts.value="";updateBatchCount();batchPrompts.focus()}

function addCard(x,first=true){
  $("empty")?.remove();
  const card=document.createElement("article");card.className="card";
  const img=document.createElement("img");img.loading="lazy";img.src=`${x.url}?t=${x.time}`;
  const info=document.createElement("div");info.className="info";
  const meta=document.createElement("div");meta.className="meta";meta.textContent=`#${x.id} · GPU${x.gpu} · ${x.seconds}s · seed ${x.seed}`;
  const p=document.createElement("div");p.className="prompt";p.textContent=x.prompt;
  info.append(meta,p);card.append(img,info);first?grid.prepend(card):grid.append(card);
}

async function status(){try{const x=await fetch("/api/status").then(r=>r.json());$("q").textContent=x.queued;$("queueCount").textContent=x.queued;$("imageCount").textContent=x.images}catch{}}
fetch("/api/history").then(r=>r.json()).then(xs=>xs.forEach(x=>addCard(x,false))).catch(()=>{});
setInterval(status,1500);status();updateBatchCount();

const proto=location.protocol==="https:"?"wss":"ws";
function connect(){
  const s=new WebSocket(`${proto}://${location.host}/ws`);
  s.onopen=()=>{$("conn").classList.add("online");$("conn").innerHTML='<span class="dot"></span>Online'};
  s.onmessage=e=>{addCard(JSON.parse(e.data));status()};
  s.onclose=()=>{$("conn").classList.remove("online");$("conn").innerHTML='<span class="dot"></span>Offline';setTimeout(connect,1500)};
}
connect();

document.addEventListener("keydown",e=>{
  if(!(e.ctrlKey||e.metaKey)||e.key!=="Enter")return;
  e.preventDefault();
  if(document.activeElement===batchPrompts)submitBatch();else submitSingle();
});
</script>
</body>
</html>''')


app.mount("/images", StaticFiles(directory=ROOT), name="images")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30100)
