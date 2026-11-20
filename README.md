# motion-engine

Troca as pessoas de um clipe longo (20 a 40 min) por personagens criados do zero
(avatar aproximado + voz clonada/gerada), com **teto de gasto por job e por dia**.

Escopo (cenário A): 1080p final, até 3 pessoas, render interno em 540p (padrão; 720p opcional) + upscale
Real-ESRGAN, só os trechos com pessoa passam pela GPU, GPU alugada (Vast.ai/RunPod).

```
vídeo ─► preparar ─► analisar ─► personagens ─► render ─► áudio ─► pós ─► output.mp4
         (ffmpeg)   YOLO+ByteTrack TripoSR+rig   Blender   Wav2Lip  Real-ESRGAN
                    + DWPose      XTTS/F5-TTS   blocos 30s         + mux FFmpeg
```

## Estrutura

```
api/            FastAPI: rotas, schemas (contrato), budget.py (tetos), costs.py (modelo de custo)
taskqueue/      Celery: celery_app.py, tasks.py (enfileira/roda o job)
workers/
  analyze/      track.py (YOLO+ByteTrack), pose.py (DWPose), blockdata.py, assign.py (track→personagem)
  character/    build_avatar.py + blender_rig.py (TripoSR → malha com esqueleto), voice.py (XTTS/F5)
  retarget/     retarget.py (pose 2D → ossos do rig, sem expressão)
  render/       render.py + blender_render.py (Blender headless), checkpoint.py (retomada por bloco)
  audio/        lipsync.py (Wav2Lip no rosto renderizado)
  post/         upscale.py (Real-ESRGAN 720→1080), mux.py (FFmpeg)
  pipeline.py   as etapas acima, com reserva de custo antes de cada uma
storage/s3.py   S3/R2 + lifecycle (tmp/ expira sozinho)
db/             models.py (SQLAlchemy), session.py, migrations/ (Alembic)
scripts/        cost_estimate.py, seed.py, download_models.sh
```

Diferenças em relação ao esboço original:

- **`queue/` virou `taskqueue/`**: `queue` é módulo da biblioteca padrão do Python e o
  Celery o importa; uma pasta com esse nome na raiz quebra o worker.
- **O teto reserva antes de gastar.** O esboço cobrava depois e checava
  `gasto_hoje < teto`, o que deixa passar um bloco inteiro além do teto, e dois workers
  juntos podiam furar o limite. Aqui cada etapa/bloco reserva o custo previsto com
  `SELECT ... FOR UPDATE` no job e no dia, e só roda se couber. Depois acerta pelo
  tempo de GPU medido (devolve se gastou menos).
- **O teto diário do request nunca passa do teto do servidor** (`DAILY_BUDGET_USD`).
  Senão qualquer cliente poderia mandar `daily_budget_usd: 99999`.
- `JobStatus.state` ganhou `cancelled`.
- `JobIn` ganhou dois campos opcionais: `duration_s` (senão a API mede com ffprobe) e
  `script` (falas `{character, start_s, text}`). **Sem `script`, o áudio original é
  mantido e o lipsync é pulado**, porque não há texto para a voz falar.

## Rodando

```bash
cp .env.example .env
docker compose up -d db redis minio api
docker compose run --rm api python -m scripts.seed      # migrations + bucket + lifecycle
bash scripts/download_models.sh models                  # pesos DWPose/Real-ESRGAN (+ Wav2Lip manual)
docker compose up -d worker                             # precisa de GPU NVIDIA
```

Na GPU alugada, rode só o worker (`docker/Dockerfile.worker`) com `DATABASE_URL`,
`REDIS_URL` e `S3_*` apontando para o servidor onde estão API, Postgres e Redis.
Um worker processa um job por vez (`--concurrency 1`).

### Interface de teste

Com a API no ar, abra **http://localhost:8000/ui** (a raiz `/` redireciona para lá).
Envie o vídeo de referência e a foto de cada pessoa (voz é opcional), clique em
**Gerar vídeo** e acompanhe estado, blocos prontos, avatares e gasto (atualiza a cada 3 s).
Quando termina, o vídeo toca na própria página e o botão **Baixar MP4** salva o arquivo.
Dá para cancelar, retomar (com teto maior, se parou por orçamento) e reabrir jobs recentes.

Sem login e sem pagamento: é só para testar o motor. O vídeo passa pela API
(`/ui/jobs/{id}/video`) porque a URL do S3 aponta para o endpoint interno (`minio:9000`),
que o navegador não alcança. O upload (`POST /ui/upload`) aceita vídeo, imagem e áudio e,
para vídeo, já mede a duração com ffprobe.

### Fluxo pela API

```bash
# 1. sobe as folhas (e amostras de voz)
curl -F file=@ana.png localhost:8000/characters/upload     # -> {"url": "s3://motion-engine/uploads/..."}

# 2. quanto vai custar?
curl -XPOST localhost:8000/jobs/estimate -H 'content-type: application/json' \
     -d '{"duration_s": 1800, "people": 3}'

# 3. cria o job (402 se a estimativa não couber no teto do job ou no saldo do dia)
curl -XPOST localhost:8000/jobs -H 'content-type: application/json' -d '{
  "video_url": "s3://motion-engine/uploads/clipe.mp4",
  "budget_usd": 30, "daily_budget_usd": 90,
  "characters": [
    {"name": "Ana",  "sheet_url": "s3://...", "voice_ref_url": "s3://.../ana.wav"},
    {"name": "Bia",  "sheet_url": "s3://...", "voice_mode": "generate"},
    {"name": "Caio", "sheet_url": "s3://..."}
  ]
}'

# 4. acompanha / corrige o mapa track -> personagem / cancela / retoma
curl localhost:8000/jobs/<id>
curl localhost:8000/characters?job_id=<id>
curl -XPUT localhost:8000/characters/<char_id>/tracks -d '{"track_ids":[3,17]}' -H 'content-type: application/json'
curl -XPOST localhost:8000/jobs/<id>/cancel
curl -XPOST localhost:8000/jobs/<id>/resume -d '{"budget_usd": 40}' -H 'content-type: application/json'
```

## Teto de gasto

| Onde | O que faz |
|---|---|
| `POST /jobs` | Estima com a duração do vídeo; 402 se não couber no teto do job ou no que sobra do dia |
| `prepare` (worker) | Refaz a estimativa com a duração real do arquivo antes de gastar GPU |
| Cada etapa/bloco | `reserve()` do custo previsto; se não cabe → `paused_budget` |
| Depois da etapa | `settle()` com o tempo real de GPU (pode devolver) |

O pior caso de estouro é **um bloco**: se a estimativa de um bloco for menor que o real,
o acerto entra depois dele e o próximo bloco já não passa. Um job em `paused_budget`
volta com `POST /jobs/{id}/resume` (com `budget_usd` maior, ou no dia seguinte para o
teto diário) e continua do bloco onde parou.

Livro-razão em `spend` (cada reserva e acerto) e total do dia em `daily_spend`.

## Blocos de 30 s, checkpoint e retomada

O vídeo é dividido em blocos de `BLOCK_SECONDS`. Cada bloco guarda a última etapa
concluída (`pending → rendered → lipsynced → upscaled`, ou `skipped` se não tem pessoa)
e a chave S3 do arquivo. Retomar (worker caiu, teto aumentado, `resume`) faz:

1. confere no S3 se os arquivos das etapas marcadas existem; se sumiu, o bloco volta uma etapa;
2. pula o que já está feito.

A análise também é salva por bloco. Como o ByteTrack precisa de continuidade, uma
retomada no meio da análise reconecta os IDs novos aos antigos por IoU.
Celery está com `acks_late`: se a máquina alugada morrer, o job volta para a fila e
outro worker continua do checkpoint. Uma trava no Redis impede dois workers no mesmo job.

Blocos sem pessoa (`process_only_with_people`) não passam pela GPU: o trecho original
é só redimensionado para a resolução final.

## Estimativa de custo

```bash
python -m scripts.cost_estimate --minutes 30 --people 3 --budget 30
python -m scripts.cost_estimate --video clipe.mp4 --people 3 --json
```

Sai com código 2 se não couber no `--budget`. O mesmo modelo (`api/costs.py`) é usado
pela API e pelos workers.

**Calibração.** Os GPU-segundos por frame em `api/costs.py` são chutes de referência
para A100 e ainda não foram medidos neste código. Com eles, 30 min / 3 pessoas / 1080p
dá ~US$ 8 (o seu esboço, com 0,4 s/frame/pessoa, dava ~US$ 26). Rode um clipe de 2 a 3
min, veja `spend` (reserva x acerto por etapa) e ajuste com `COST_*` no `.env`. Até
calibrar, use tetos conservadores.

## Limitações conhecidas (avatar aproximado)

- **Retarget 2D**: os ossos giram só no plano da imagem (câmera ortográfica). Braço
  apontando para a câmera vira braço curto; não há expressão facial.
- **Inpainting** (`workers/inpaint/clean.py`): antes do render, a pessoa que vira
  avatar é apagada (máscara YOLO-seg + fundo pela mediana do bloco quando a câmera
  está parada; LaMa quadro a quadro quando a câmera mexe). Pessoas que não viram
  personagem ficam. Desligue com `INPAINT=false`.
- **Folha de referência**: usa o painel mais à esquerda como vista frontal
  (`SHEET_PANELS` força o número de painéis). O rig supõe A-pose; se a malha sair
  deitada ou de costas, ajuste `AVATAR_EULER`.
- **Mapa track → personagem automático** usa continuidade de tempo/posição, sem
  reconhecer aparência. Em cortes de câmera pode trocar personagens; corrija com
  `PUT /characters/{id}/tracks`.
- **Dublagem**: com `script`, as vozes entram por cima do áudio original abaixado
  (-14 dB). Não separa a fala original da música (Demucs entra em `mix_dub`).
- Wav2Lip precisa de `wav2lip_gan.pth` baixado à mão.

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Cobrem tetos (job, dia, concorrência de jobs, acerto), checkpoint/retomada, modelo de
custo e CLI, API, retarget, mapa de tracks e o caminho FFmpeg (corte, passthrough,
concat, mux). As partes de GPU (YOLO, DWPose, TripoSR, Blender, Wav2Lip, Real-ESRGAN)
não rodam no CI e não foram executadas ainda; o primeiro teste real é um clipe curto
no worker de GPU.
