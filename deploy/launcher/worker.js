// Página "Ligar / Desligar" do Motion Engine (Cloudflare Worker).
// Link fixo: /abrir leva sempre para a interface do pod que estiver ligado.
// Ligar: tenta religar o pod existente; se a GPU dele foi alugada por outro,
// apaga esse pod (o disco fixo NÃO é apagado) e cria outro no mesmo disco com
// a primeira GPU livre da lista GPU_TYPES.

const API = "https://rest.runpod.io/v1";

async function rp(env, path, init = {}) {
  const r = await fetch(API + path, {
    ...init,
    headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}`, "Content-Type": "application/json" },
  });
  const text = await r.text();
  let body; try { body = JSON.parse(text); } catch { body = text; }
  if (!r.ok) throw new Error(`RunPod ${r.status}: ${typeof body === "string" ? body : JSON.stringify(body)}`);
  return body;
}

const uiUrl = (id) => `https://${id}-8000.proxy.runpod.net/ui`;

async function ourPods(env) {
  const pods = await rp(env, "/pods");
  return (Array.isArray(pods) ? pods : pods.pods || []).filter((p) => p.name === env.POD_NAME);
}

async function ready(id) {
  try {
    const r = await fetch(`https://${id}-8000.proxy.runpod.net/health`, { signal: AbortSignal.timeout(5000) });
    return r.ok;
  } catch { return false; }
}

async function status(env) {
  const pods = await ourPods(env);
  const running = pods.find((p) => p.desiredStatus === "RUNNING");
  if (!running) return { state: "off" };
  const ok = await ready(running.id);
  return { state: ok ? "ready" : "starting", id: running.id, url: uiUrl(running.id),
           gpu: running.machine?.gpuTypeId || running.gpu?.displayName || null,
           cost: running.costPerHr ?? null };
}

async function create(env) {
  const body = {
    name: env.POD_NAME,
    imageName: env.IMAGE,
    gpuTypeIds: env.GPU_TYPES.split(",").map((s) => s.trim()).filter(Boolean),
    gpuTypePriority: "custom",
    gpuCount: 1,
    cloudType: "SECURE",
    networkVolumeId: env.NETWORK_VOLUME_ID,
    volumeMountPath: "/workspace",
    containerDiskInGb: Number(env.CONTAINER_DISK_GB || 30),
    ports: ["8000/http", "22/tcp"],
    dockerStartCmd: ["bash", "-c", env.START_CMD],
  };
  return rp(env, "/pods", { method: "POST", body: JSON.stringify(body) });
}

async function start(env) {
  const pods = await ourPods(env);
  const running = pods.find((p) => p.desiredStatus === "RUNNING");
  if (running) return { id: running.id, note: "já estava ligado" };
  for (const p of pods) {
    try {
      await rp(env, `/pods/${p.id}/start`, { method: "POST" });
      return { id: p.id, note: "religado" };
    } catch (e) {
      // GPU dele indisponível: apaga só o pod (o Network Volume fica intacto)
      await rp(env, `/pods/${p.id}`, { method: "DELETE" }).catch(() => {});
    }
  }
  const pod = await create(env);
  return { id: pod.id, note: "pod novo numa GPU livre" };
}

async function stop(env) {
  const pods = await ourPods(env);
  for (const p of pods.filter((p) => p.desiredStatus === "RUNNING")) {
    await rp(env, `/pods/${p.id}/stop`, { method: "POST" });
  }
  return { stopped: true };
}

// ---- senha simples (cookie assinado) ------------------------------------------
async function sign(env, v) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(env.PASSWORD),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const s = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(v));
  return btoa(String.fromCharCode(...new Uint8Array(s))).replace(/[^a-zA-Z0-9]/g, "");
}
async function authed(env, req) {
  const c = (req.headers.get("Cookie") || "").match(/me_auth=([A-Za-z0-9]+)/);
  return !!c && c[1] === (await sign(env, "ok"));
}

const json = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { "Content-Type": "application/json" } });

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname === "/login" && req.method === "POST") {
      const form = await req.formData();
      if (form.get("senha") !== env.PASSWORD) return new Response(PAGE(true), { status: 401, headers: { "Content-Type": "text/html; charset=utf-8" } });
      return new Response(null, { status: 303, headers: { Location: "/",
        "Set-Cookie": `me_auth=${await sign(env, "ok")}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000` } });
    }
    if (!(await authed(env, req))) return new Response(PAGE(false), { headers: { "Content-Type": "text/html; charset=utf-8" } });

    try {
      if (url.pathname === "/api/status") return json(await status(env));
      if (url.pathname === "/api/start" && req.method === "POST") return json(await start(env));
      if (url.pathname === "/api/stop" && req.method === "POST") return json(await stop(env));
      if (url.pathname === "/abrir") {
        const s = await status(env);
        return s.state === "ready" ? Response.redirect(s.url, 302) : Response.redirect(url.origin + "/", 302);
      }
    } catch (e) {
      return json({ error: String(e.message || e) }, 502);
    }
    return new Response(PANEL, { headers: { "Content-Type": "text/html; charset=utf-8" } });
  },
};

const STYLE = `<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e3de;--accent:#2f5bea;--ok:#1f8a4c;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--card:#1f1f1d;--ink:#ecece8;--muted:#9a9a93;--line:#33332f;--accent:#6b8cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:460px;margin:0 auto;padding:28px 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
h1{font-size:22px;margin:0 0 6px}p{color:var(--muted);margin:0 0 16px}
button,input{font:inherit;width:100%;min-height:50px;border-radius:10px;border:1px solid var(--line);padding:0 14px;margin-top:10px;background:transparent;color:inherit}
button.p{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}button:disabled{opacity:.5}
.st{font-weight:600;font-size:18px;margin:6px 0 2px}.ok{color:var(--ok)}.bad{color:var(--bad)}.hint{font-size:14px;color:var(--muted)}
</style>`;

const PAGE = (err) => `<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Motion Engine</title>${STYLE}</head>
<body><main><div class="card"><h1>Motion Engine</h1><p>Digite a senha para ligar ou desligar.</p>
<form method="post" action="/login"><input type="password" name="senha" placeholder="Senha" autofocus>
${err ? '<div class="bad" style="margin-top:8px">Senha errada.</div>' : ""}<button class="p">Entrar</button></form></div></main></body></html>`;

const PANEL = `<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Motion Engine</title>${STYLE}</head>
<body><main><div class="card"><h1>Motion Engine</h1><p>Ligue antes de usar. Ele desliga sozinho depois de 15 min parado.</p>
<div class="st" id="st">Verificando…</div><div class="hint" id="sub"></div>
<button class="p" id="open" style="display:none">Abrir o Motion Engine</button>
<button class="p" id="on">Ligar</button><button id="off">Desligar</button></div></main>
<script>
const $=id=>document.getElementById(id);let url=null,t=null;
async function api(p,m){const r=await fetch(p,{method:m||"GET"});const j=await r.json();if(j.error)throw new Error(j.error);return j}
function show(s){url=s.url||null;$("open").style.display=s.state==="ready"?"":"none";$("on").style.display=s.state==="off"?"":"none";$("off").style.display=s.state==="off"?"none":"";
 $("st").textContent={off:"Desligado",starting:"Ligando… (3 a 5 min)",ready:"Ligado e pronto"}[s.state];$("st").className="st "+(s.state==="ready"?"ok":"");
 $("sub").textContent=s.state==="off"?"Sem custo de GPU agora.":(s.cost?"Custando US$ "+s.cost+"/h enquanto estiver ligado.":"")}
async function poll(){try{show(await api("/api/status"))}catch(e){$("st").textContent="Erro";$("sub").textContent=e.message}clearTimeout(t);t=setTimeout(poll,8000)}
$("on").onclick=async()=>{$("on").disabled=true;$("st").textContent="Procurando uma GPU livre…";try{await api("/api/start","POST")}catch(e){$("sub").textContent=e.message}$("on").disabled=false;poll()};
$("off").onclick=async()=>{$("off").disabled=true;try{await api("/api/stop","POST")}catch(e){$("sub").textContent=e.message}$("off").disabled=false;poll()};
$("open").onclick=()=>{if(url)location.href=url};poll();
</script></body></html>`;
