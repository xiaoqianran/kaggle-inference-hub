import json,time,asyncio,hashlib,queue,itertools
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI,UploadFile,File,Form,Header,HTTPException,WebSocket
from fastapi.responses import HTMLResponse,Response
from fastapi.staticfiles import StaticFiles
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

app=FastAPI(); ROOT=Path("sana_received"); ROOT.mkdir(exist_ok=True)
PASSWORD="wangran"; KEY=hashlib.sha256(PASSWORD.encode()).digest(); clients=set(); history=[]; task_queue=queue.Queue(); task_ids=itertools.count(1)

# 定义单个生成任务
class TaskIn(BaseModel):
    prompt:str
    seed:int|None=None
    width:int=1024
    height:int=1024
    steps:int=2

# 定义批量生成任务
class BatchIn(BaseModel):
    prompts:list[str]
    seed:int|None=None
    width:int=1024
    height:int=1024
    steps:int=2

# 校验访问密码
def auth(authorization):
    if authorization!=f"Bearer {PASSWORD}": raise HTTPException(401,"Unauthorized")

# 创建生成任务
def make_task(prompt,width,height,steps,seed=None):
    return {"id":next(task_ids),"prompt":prompt,"seed":seed if seed is not None else int(time.time()*1000000)%2147483647,"width":width,"height":height,"steps":steps}

# AES-GCM 解密图片
def decrypt(data):
    return AESGCM(KEY).decrypt(data[:12],data[12:],None)

# 向网页广播生成结果
async def broadcast(item):
    dead=[]
    for ws in list(clients):
        try: await ws.send_text(json.dumps(item,ensure_ascii=False))
        except Exception: dead.append(ws)
    for ws in dead: clients.discard(ws)

# 提交单个 Prompt
@app.post("/task")
def add_task(x:TaskIn,authorization:str|None=Header(None)):
    auth(authorization); task=make_task(x.prompt,x.width,x.height,x.steps,x.seed); task_queue.put(task); return {"queued":True,**task}

# 批量提交 Prompt
@app.post("/task/batch")
def add_batch(x:BatchIn,authorization:str|None=Header(None)):
    auth(authorization); prompts=[p.strip() for p in x.prompts if p.strip()]; base=x.seed
    items=[]
    for i,p in enumerate(prompts):
        task=make_task(p,x.width,x.height,x.steps,base+i if base is not None else None); task_queue.put(task); items.append(task)
    return {"queued":len(items),"queue_size":task_queue.qsize(),"tasks":items}

# Kaggle 长轮询领取任务
@app.get("/task/next")
async def next_task(authorization:str|None=Header(None)):
    auth(authorization)
    try: return await asyncio.to_thread(task_queue.get,True,25)
    except queue.Empty: return Response(status_code=204)

# 返回运行状态
@app.get("/api/status")
def status(): return {"queued":task_queue.qsize(),"images":len(history),"clients":len(clients)}

# 接收 Kaggle 加密图片
@app.post("/upload")
async def upload(file:UploadFile=File(...),id:int=Form(...),gpu:int=Form(...),seed:int=Form(...),prompt:str=Form(""),seconds:float=Form(0),authorization:str|None=Header(None)):
    auth(authorization)
    try: data=await asyncio.to_thread(decrypt,await file.read())
    except Exception: raise HTTPException(400,"Decrypt failed")
    name=f"{int(time.time()*1000)}_{id:04d}_gpu{gpu}_seed{seed}.webp"; await asyncio.to_thread((ROOT/name).write_bytes,data)
    item={"id":id,"gpu":gpu,"seed":seed,"prompt":prompt,"seconds":seconds,"url":f"/images/{name}","time":time.time()}; history.append(item); del history[:-300]; await broadcast(item); return item

# 返回历史图片
@app.get("/api/history")
def get_history(): return history[-200:]

# 建立实时 WebSocket
@app.websocket("/ws")
async def ws(websocket:WebSocket):
    await websocket.accept(); clients.add(websocket)
    try:
        while True: await websocket.receive_text()
    except Exception: clients.discard(websocket)

# 返回控制台网页
@app.get("/",response_class=HTMLResponse)
def index():
    return HTMLResponse(r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SANA Control</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#11111b;color:#cdd6f4;font:14px Inter,system-ui,sans-serif}.shell{max-width:1600px;margin:auto;padding:20px}.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}.title{font-size:22px;font-weight:800}.sub{color:#7f849c;font-size:12px;margin-top:4px}.status{display:flex;gap:8px;align-items:center}.pill{background:#1e1e2e;border:1px solid #313244;border-radius:999px;padding:7px 11px;color:#bac2de}.dot{width:8px;height:8px;border-radius:50%;background:#f38ba8;display:inline-block;margin-right:6px}.online .dot{background:#a6e3a1}.layout{display:grid;grid-template-columns:390px 1fr;gap:18px}.panel{background:#181825;border:1px solid #313244;border-radius:16px;padding:18px;box-shadow:0 15px 40px #0003}.panel h2{font-size:14px;margin:0 0 14px;color:#cba6f7}.label{font-size:11px;color:#9399b2;margin:12px 0 6px}textarea,input,select{width:100%;background:#11111b;color:#cdd6f4;border:1px solid #313244;border-radius:10px;padding:10px;outline:none}textarea{height:260px;resize:vertical;line-height:1.55}textarea:focus,input:focus{border-color:#89b4fa}.row{display:grid;grid-template-columns:1fr 1fr;gap:9px}.actions{display:flex;gap:8px;margin-top:14px}.btn{border:0;border-radius:10px;padding:10px 14px;font-weight:700;cursor:pointer}.primary{background:#89b4fa;color:#11111b;flex:1}.ghost{background:#313244;color:#cdd6f4}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px}.stat{background:#11111b;border-radius:10px;padding:10px}.stat b{display:block;font-size:17px}.stat span{color:#7f849c;font-size:10px}.gallery-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.gallery{columns:4 240px;column-gap:12px}.card{break-inside:avoid;margin-bottom:12px;background:#1e1e2e;border:1px solid #313244;border-radius:13px;overflow:hidden;animation:in .2s ease}.card img{width:100%;display:block}.info{padding:10px}.meta{font-size:10px;color:#89b4fa}.prompt{font-size:12px;color:#bac2de;margin-top:5px;line-height:1.4}.empty{color:#585b70;text-align:center;padding:80px 20px}.toast{position:fixed;right:20px;bottom:20px;background:#a6e3a1;color:#11111b;padding:11px 15px;border-radius:10px;font-weight:700;display:none;z-index:5}@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}@media(max-width:900px){.layout{grid-template-columns:1fr}.panel{position:static}.gallery{columns:2 180px}}
</style></head>
<body><div class="shell">
<div class="top"><div><div class="title">SANA Control</div><div class="sub">Local Control Plane · Kaggle Dual GPU</div></div><div class="status"><span class="pill" id="conn"><span class="dot"></span>Connecting</span><span class="pill">Queue <b id="q">0</b></span></div></div>
<div class="layout">
<section class="panel">
<h2>GENERATE</h2>
<div class="label">ACCESS TOKEN</div><input id="token" type="password" placeholder="wangran">
<div class="label">PROMPTS · 一行一个</div><textarea id="prompts" placeholder="A cinematic mountain lake at sunrise&#10;A futuristic Tokyo street at night&#10;A peaceful forest covered in mist"></textarea>
<div class="row"><div><div class="label">WIDTH</div><input id="width" type="number" value="1024"></div><div><div class="label">HEIGHT</div><input id="height" type="number" value="1024"></div></div>
<div class="row"><div><div class="label">STEPS</div><input id="steps" type="number" value="2"></div><div><div class="label">BASE SEED · 可空</div><input id="seed" type="number" placeholder="随机"></div></div>
<div class="actions"><button class="btn primary" onclick="submitTasks()">加入生成队列</button><button class="btn ghost" onclick="clearPrompts()">清空</button></div>
<div class="stats"><div class="stat"><b id="promptCount">0</b><span>PROMPTS</span></div><div class="stat"><b id="queueCount">0</b><span>QUEUED</span></div><div class="stat"><b id="imageCount">0</b><span>IMAGES</span></div></div>
</section>
<section><div class="gallery-head"><div><b>Live Results</b><div class="sub">生成完成后自动出现</div></div><button class="btn ghost" onclick="location.reload()">刷新</button></div><main id="grid" class="gallery"><div class="empty" id="empty">等待第一张图片...</div></main></section>
</div></div><div class="toast" id="toast"></div>
<script>
const $=x=>document.getElementById(x),grid=$("grid"),token=$("token"),prompts=$("prompts"); token.value=localStorage.getItem("sana_token")||"";
const lines=()=>prompts.value.split("\n").map(x=>x.trim()).filter(Boolean); prompts.oninput=()=>$("promptCount").textContent=lines().length; token.onchange=()=>localStorage.setItem("sana_token",token.value);
function toast(t,bad=false){const x=$("toast");x.textContent=t;x.style.background=bad?"#f38ba8":"#a6e3a1";x.style.display="block";setTimeout(()=>x.style.display="none",2200)}
function clearPrompts(){prompts.value="";prompts.oninput()}
function add(x,first=true){$("empty")?.remove();const c=document.createElement("article");c.className="card";c.innerHTML=`<img loading="lazy" src="${x.url}?t=${x.time}"><div class="info"><div class="meta">#${x.id} · GPU${x.gpu} · ${x.seconds}s · seed ${x.seed}</div><div class="prompt"></div></div>`;c.querySelector(".prompt").textContent=x.prompt;first?grid.prepend(c):grid.append(c)}
async function submitTasks(){const ps=lines();if(!ps.length)return toast("请输入 Prompt",true);const body={prompts:ps,width:+$("width").value,height:+$("height").value,steps:+$("steps").value};if($("seed").value)body.seed=+$("seed").value;
try{const r=await fetch("/task/batch",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+token.value},body:JSON.stringify(body)});if(!r.ok)throw new Error(await r.text());const x=await r.json();toast(`已加入 ${x.queued} 个任务`);prompts.value="";prompts.oninput();status()}catch(e){toast("提交失败："+e.message,true)}}
async function status(){try{const x=await fetch("/api/status").then(r=>r.json());$("q").textContent=x.queued;$("queueCount").textContent=x.queued;$("imageCount").textContent=x.images}catch{}}
fetch("/api/history").then(r=>r.json()).then(xs=>xs.forEach(x=>add(x,false)));setInterval(status,1500);status();
const proto=location.protocol==="https:"?"wss":"ws";function connect(){const s=new WebSocket(`${proto}://${location.host}/ws`);s.onopen=()=>{$("conn").classList.add("online");$("conn").innerHTML='<span class="dot"></span>Online'};s.onmessage=e=>{add(JSON.parse(e.data));status()};s.onclose=()=>{$("conn").classList.remove("online");$("conn").innerHTML='<span class="dot"></span>Offline';setTimeout(connect,1500)}}connect();
document.addEventListener("keydown",e=>{if((e.ctrlKey||e.metaKey)&&e.key==="Enter")submitTasks()});
</script></body></html>""")

app.mount("/images",StaticFiles(directory=ROOT),name="images")

if __name__=="__main__": import uvicorn; uvicorn.run(app,host="0.0.0.0",port=30100)
