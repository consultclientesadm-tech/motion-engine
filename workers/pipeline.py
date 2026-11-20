"""Pipeline do job: preparar -> analisar -> personagens -> render -> áudio -> pós.

Regras que valem em todas as etapas:
  * antes de qualquer trabalho de GPU, `spend()` reserva o custo previsto
    (teto do job e do dia); se não couber, BudgetExceeded -> paused_budget;
  * depois, acerta previsto x medido;
  * entre blocos, confere se o job foi cancelado e renova a trava do Redis;
  * o resultado de cada bloco vai para o S3 antes de marcar o checkpoint.
"""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from api import costs
from api.budget import reserve, settle
from db.models import Block, BlockStage, Character, Job, JobState
from db.session import session_scope
from settings import get_settings
from storage import s3
from workers.media import GpuTimer, cut, extract_audio, normalize_source, probe
from workers.render import checkpoint


class Cancelled(Exception):
    pass


@dataclass
class Ctx:
    job_id: str
    lock: object | None = None
    _dir: Path | None = None
    _cache: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        if self._dir is None:
            self._dir = Path(get_settings().work_dir) / self.job_id
            self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir

    def sub(self, name: str) -> Path:
        d = self.dir / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        return d

    def local(self, key: str, name: str | None = None) -> Path:
        """Baixa uma chave do S3 uma vez por execução."""
        if key not in self._cache:
            self._cache[key] = s3.fetch(key, self.dir / "cache" / (name or key.replace("/", "_")))
        return self._cache[key]

    def heartbeat(self) -> None:
        if self.lock is not None:
            self.lock.reacquire()
        with session_scope() as s:
            if s.get(Job, self.job_id).state == JobState.cancelled:
                raise Cancelled()

    def cleanup(self) -> None:
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)


# ---- utilidades ------------------------------------------------------------------

def set_state(job_id: str, state: JobState, error: str | None = None) -> None:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job.state == JobState.cancelled and state != JobState.cancelled:
            return
        job.state = state
        if error is not None:
            job.error = error


@contextmanager
def spend(job_id: str, stage: str, usd: float, *, cpu: bool = False):
    """Reserva -> executa -> acerta pelo tempo medido."""
    with session_scope() as s:
        reserve(s, job_id, stage, usd)
    timer = GpuTimer()
    try:
        with timer:
            yield
    finally:
        actual = costs.cpu_usd(timer.seconds) if cpu else costs.gpu_usd(timer.seconds)
        with session_scope() as s:
            settle(s, job_id, stage, usd, actual)


def load_job(session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    _ = job.blocks, job.characters  # carrega relações antes de fechar a sessão
    return job


# ---- 0. preparar -------------------------------------------------------------------

def prepare(ctx: Ctx) -> None:
    st = get_settings()
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        if job.source_key and job.duration_s and job.blocks:
            return
        video_url = job.video_url

    raw = s3.fetch(video_url, ctx.dir / "raw_input")
    info = probe(raw)
    src = ctx.dir / "source.mp4"
    normalize_source(raw, src, st.fps)
    raw.unlink(missing_ok=True)
    info = probe(src)
    source_key = s3.upload(src, s3.job_key(ctx.job_id, "source.mp4"), "video/mp4")
    ctx._cache[source_key] = src
    audio_key = None
    if info.has_audio:
        wav = ctx.dir / "original.wav"
        extract_audio(src, wav)
        audio_key = s3.upload(wav, s3.job_key(ctx.job_id, "audio", "original.wav"), "audio/wav")

    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        job.source_key, job.audio_key = source_key, audio_key
        job.duration_s, job.fps = info.duration_s, st.fps
        job.width, job.height, job.has_audio = info.width, info.height, info.has_audio
        # com a duração real, refaz a estimativa: não começa a gastar se não couber
        est = estimate_for(job)
        job.estimated_usd = est.total_usd
        budget = job.budget_usd
        checkpoint.ensure_blocks(s, job, st.block_seconds)
    if est.total_usd > budget:
        from api.budget import BudgetExceeded
        raise BudgetExceeded("do job (estimativa com a duração real)", est.total_usd, budget)


def estimate_for(job: Job) -> costs.Estimate:
    speech = sum(len(ln["text"]) for ln in (job.script or [])) / 15.0  # ~15 caracteres/s
    return costs.estimate(job.duration_s, people=len(job.characters), fps=job.fps or get_settings().fps,
                          target_resolution=job.target_resolution,
                          only_people=job.process_only_with_people,
                          speech_s=speech, lipsync=bool(job.script))


# ---- 1. analisar (YOLO + ByteTrack + DWPose) -------------------------------------

def analyze(ctx: Ctx) -> None:
    from workers.analyze.assign import SummaryBuilder, TrackSummary, auto_assign
    from workers.analyze.blockdata import BlockAnalysis
    from workers.analyze.pose import PoseEstimator
    from workers.analyze.track import Tracker, run_tracking

    st = get_settings()
    set_state(ctx.job_id, JobState.analyzing)
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        todo = [(b.index, b.start_s, b.end_s) for b in job.blocks if not b.analysis_key]
        done_keys = {b.index: b.analysis_key for b in job.blocks if b.analysis_key}
        fps, n_chars, w = job.fps, len(job.characters), job.width
        source_key, tracks_key = job.source_key, job.tracks_key
        only_people = job.process_only_with_people

    summary = SummaryBuilder.from_json(s3.get_json(tracks_key)) if tracks_key and s3.exists(tracks_key) \
        else SummaryBuilder()
    if todo:
        src = ctx.local(source_key, "source.mp4")
        tracker = Tracker()
        pose = PoseEstimator()
        prev_idx = todo[0][0] - 1
        if prev_idx in done_keys:  # retomada no meio: costura IDs
            prev = BlockAnalysis.from_bytes(s3.client().get_object(
                Bucket=s3.bucket(), Key=done_keys[prev_idx])["Body"].read())
            tracker.id_offset = (max(summary.tracks) + 1000) if summary.tracks else 0
            stitch_from = prev.last_boxes()
        else:
            stitch_from = None

        for idx, a, b in todo:
            ctx.heartbeat()
            f0, f1 = int(round(a * fps)), int(round(b * fps))
            usd = costs.gpu_usd(costs.analyze_s(b - a, fps, n_chars, 1.0))
            rows = []
            with spend(ctx.job_id, f"analyze:{idx}", usd):
                for img, ft in run_tracking(src, f0, f1, tracker):
                    if stitch_from is not None and len(ft.ids):
                        m = tracker.stitch(stitch_from, ft)
                        ft.ids = np.array([m.get(int(i), int(i)) for i in ft.ids])
                        stitch_from = None
                    if not len(ft.ids):
                        continue
                    kp = pose(img, ft.boxes)
                    for tid, box, k in zip(ft.ids, ft.boxes, kp):
                        rows.append((ft.frame, int(tid), box, k))
                        summary.add(ft.frame, int(tid), box)
            ba = BlockAnalysis.from_rows(rows, f0, f1)
            key = s3.upload_bytes(ba.to_bytes(), s3.job_key(ctx.job_id, "analysis", f"block_{idx:04d}.npz"))
            tracks_key = s3.put_json(summary.to_json(), s3.job_key(ctx.job_id, "analysis", "tracks.json"))
            with session_scope() as s:
                blk = s.query(Block).filter_by(job_id=ctx.job_id, index=idx).one()
                blk.analysis_key = key
                blk.people = ba.max_people()
                blk.has_people = ba.presence_ratio() >= st.min_person_ratio
                s.get(Job, ctx.job_id).tracks_key = tracks_key

    # mapa track -> personagem (respeita o que o usuário mapeou à mão)
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        auto = [c for c in job.characters if not c.track_ids_manual]
        if auto:
            manual_ids = {t for c in job.characters if c.track_ids_manual for t in c.track_ids}
            tracks = [TrackSummary(**t.__dict__) for t in summary.tracks.values() if t.id not in manual_ids]
            mapping = auto_assign(tracks, [c.name for c in auto], float(w), min_frames=int(fps))
            for c in auto:
                c.track_ids = mapping.get(c.name, [])
        if not only_people:
            for blk in job.blocks:
                blk.has_people = True


# ---- 2. personagens (avatar + voz) ----------------------------------------------

def build(ctx: Ctx) -> None:
    from workers.character.build_avatar import build_avatar
    set_state(ctx.job_id, JobState.building)
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        chars = [(c.id, c.name, c.sheet_url) for c in job.characters if not c.avatar_key]
    for cid, name, sheet_url in chars:
        ctx.heartbeat()
        wd = ctx.sub(f"avatar_{cid}")
        with spend(ctx.job_id, f"avatar:{name}", costs.gpu_usd(costs.build_s(1))):
            sheet = s3.fetch(sheet_url, wd / "sheet")
            glb = build_avatar(sheet, wd)
        blend_key = s3.upload(glb.with_suffix(".blend"), s3.job_key(ctx.job_id, "characters", f"{cid}.blend"))
        s3.upload(glb, s3.job_key(ctx.job_id, "characters", f"{cid}.glb"), "model/gltf-binary")
        with session_scope() as s:
            s.get(Character, cid).avatar_key = blend_key
    build_voices(ctx)


def build_voices(ctx: Ctx) -> None:
    from workers.character.voice import VoiceEngine, build_voice_track, write_wav
    from workers.post.mux import mix_dub
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        script = job.script or []
        chars = [dict(id=c.id, name=c.name, voice_mode=c.voice_mode, voice_ref_url=c.voice_ref_url,
                      voice_key=c.voice_key) for c in job.characters]
        duration, orig_audio = job.duration_s, job.audio_key
        final_audio_ready = job.audio_key and "/final" in job.audio_key
    if not script or final_audio_ready:
        return
    engine = VoiceEngine()
    for c in chars:
        lines = [ln for ln in script if ln["character"] == c["name"]]
        if c["voice_key"] or not lines:
            continue
        ctx.heartbeat()
        wd = ctx.sub(f"voice_{c['id']}")
        speech_est = sum(len(ln["text"]) for ln in lines) / 15.0
        with spend(ctx.job_id, f"voice:{c['name']}", costs.gpu_usd(costs.build_s(0, speech_est))):
            ref = s3.fetch(c["voice_ref_url"], wd / "ref.wav") if c["voice_ref_url"] else None
            track, _ = build_voice_track(engine, c, lines, duration, ref)
            wav = write_wav(wd / "voice.wav", track)
        key = s3.upload(wav, s3.job_key(ctx.job_id, "audio", f"voice_{c['id']}.wav"), "audio/wav")
        with session_scope() as s:
            s.get(Character, c["id"]).voice_key = key
    # trilha final: vozes sobre o original abaixado
    with session_scope() as s:
        voice_keys = [c.voice_key for c in load_job(s, ctx.job_id).characters if c.voice_key]
    wd = ctx.sub("mix")
    out = mix_dub(ctx.local(orig_audio, "original.wav") if orig_audio else None,
                  [ctx.local(k) for k in voice_keys], wd / "final.wav")
    key = s3.upload(out, s3.job_key(ctx.job_id, "audio", "final.wav"), "audio/wav")
    with session_scope() as s:
        s.get(Job, ctx.job_id).audio_key = key


# ---- 3. render em blocos de 30 s ------------------------------------------------

def _track_map(job: Job) -> dict[int, str]:
    return {int(t): c.name for c in job.characters for t in c.track_ids}


def render_all(ctx: Ctx) -> None:
    from workers.analyze.blockdata import BlockAnalysis
    from workers.render.render import render_block
    st = get_settings()
    set_state(ctx.job_id, JobState.rendering)
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        checkpoint.verify(job, s3.exists)
        todo = [(b.index, b.start_s, b.end_s, b.people, b.analysis_key)
                for b in checkpoint.pending(job, BlockStage.rendered) if b.has_people]
        tmap, fps = _track_map(job), job.fps
        avatars = {c.name: c.avatar_key for c in job.characters}
        source_key, w, h = job.source_key, job.width, job.height

    for idx, a, b, people, akey in todo:
        ctx.heartbeat()
        wd = ctx.sub(f"block_{idx}")
        usd = costs.gpu_usd(costs.render_block_s(b - a, fps, max(people, 1)))
        with spend(ctx.job_id, f"render:{idx}", usd):
            src = ctx.local(source_key, "source.mp4")
            src_block = wd / "src.mp4"
            cut(src, src_block, a, b, st.render_width, st.render_height)
            analysis = BlockAnalysis.from_bytes(
                s3.client().get_object(Bucket=s3.bucket(), Key=akey)["Body"].read())
            local_avatars = {n: ctx.local(k, f"{n}.blend") for n, k in avatars.items() if k}
            out, pose = render_block(analysis=analysis, track_to_char=tmap, avatars=local_avatars,
                                     source_block=src_block, src_w=w, src_h=h, fps=fps,
                                     workdir=wd, out=wd / "render.mp4")
        s3.put_json(pose, s3.tmp_key(ctx.job_id, "blocks", str(idx), "pose.json"))
        key = s3.upload(out, s3.tmp_key(ctx.job_id, "blocks", str(idx), "render.mp4"), "video/mp4")
        with session_scope() as s:
            blk = s.query(Block).filter_by(job_id=ctx.job_id, index=idx).one()
            checkpoint.mark(blk, BlockStage.rendered, key)
            s3.put_json(checkpoint.manifest(load_job(s, ctx.job_id)), s3.job_key(ctx.job_id, "checkpoint.json"))
        shutil.rmtree(wd, ignore_errors=True)


# ---- 4. áudio (lipsync) ------------------------------------------------------------

def audio_all(ctx: Ctx) -> None:
    set_state(ctx.job_id, JobState.audio)
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        todo = [(b.index, b.start_s, b.end_s, b.render_key)
                for b in checkpoint.pending(job, BlockStage.lipsynced) if b.stage == BlockStage.rendered]
        voices = {c.name: c.voice_key for c in job.characters if c.voice_key}
        fps = job.fps
    if not voices:  # sem falas geradas: mantém o render como está
        with session_scope() as s:
            for idx, *_ in todo:
                blk = s.query(Block).filter_by(job_id=ctx.job_id, index=idx).one()
                checkpoint.mark(blk, BlockStage.lipsynced, blk.render_key)
        return

    import soundfile as sf

    from workers.audio.lipsync import SR, LipSyncer, lipsync_block
    syncer = LipSyncer()
    full = {}
    for name, key in voices.items():
        wav, sr = sf.read(str(ctx.local(key)), dtype="float32")
        full[name] = np.interp(np.arange(0, len(wav), sr / SR), np.arange(len(wav)), wav).astype(np.float32)
    for idx, a, b, render_key in todo:
        ctx.heartbeat()
        wd = ctx.sub(f"lip_{idx}")
        seg = {n: w[int(a * SR):int(b * SR)] for n, w in full.items()}
        seg = {n: w for n, w in seg.items() if len(w) and float(np.abs(w).max()) > 0.01}
        pose = s3.get_json(s3.tmp_key(ctx.job_id, "blocks", str(idx), "pose.json"))
        faces = max((len(fr["actors"]) for fr in pose["frames"]), default=0)
        key = render_key
        if seg and faces:
            usd = costs.gpu_usd(costs.lipsync_block_s(b - a, fps, min(faces, len(seg))))
            with spend(ctx.job_id, f"lipsync:{idx}", usd):
                video = s3.fetch(render_key, wd / "render.mp4")
                out = lipsync_block(video=video, pose=pose, voices=seg, fps=fps,
                                    out=wd / "lipsync.mp4", syncer=syncer)
            key = s3.upload(out, s3.tmp_key(ctx.job_id, "blocks", str(idx), "lipsync.mp4"), "video/mp4")
        with session_scope() as s:
            blk = s.query(Block).filter_by(job_id=ctx.job_id, index=idx).one()
            checkpoint.mark(blk, BlockStage.lipsynced, key)
        shutil.rmtree(wd, ignore_errors=True)


# ---- 5. pós: upscale 720 -> 1080 e mux -----------------------------------------------

def post_all(ctx: Ctx) -> None:
    from workers.post.mux import mux
    from workers.post.upscale import Upscaler, passthrough
    st = get_settings()
    set_state(ctx.job_id, JobState.post)
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        checkpoint.verify(job, s3.exists)
        res = (st.output_width, st.output_height) if job.target_resolution == "1080p" \
            else (st.render_width, st.render_height)
        todo = [(b.index, b.start_s, b.end_s, b.has_people, b.stage, b.lipsync_key)
                for b in job.blocks if b.stage not in checkpoint.DONE_FOR_MUX]
        fps, source_key, target = job.fps, job.source_key, job.target_resolution

    upscaler = None
    for idx, a, b, has_people, stage, lip_key in todo:
        ctx.heartbeat()
        wd = ctx.sub(f"post_{idx}")
        if not has_people:
            with spend(ctx.job_id, f"passthrough:{idx}", costs.cpu_usd(costs.mux_cpu_s(b - a)), cpu=True):
                out = passthrough(ctx.local(source_key, "source.mp4"), wd / "final.mp4", a, b, *res, fps)
            new_stage = BlockStage.skipped
        elif stage != BlockStage.lipsynced:
            raise RuntimeError(f"bloco {idx} chegou ao pós sem render ({stage.value})")
        elif target == "1080p":
            upscaler = upscaler or Upscaler()
            with spend(ctx.job_id, f"upscale:{idx}", costs.gpu_usd(costs.upscale_block_s(b - a, fps))):
                out = upscaler.upscale_video(s3.fetch(lip_key, wd / "in.mp4"), wd / "final.mp4", fps)
            new_stage = BlockStage.upscaled
        else:
            out, new_stage = None, BlockStage.upscaled  # 720p: o bloco renderizado já é o final
        key = (s3.upload(out, s3.tmp_key(ctx.job_id, "blocks", str(idx), "final.mp4"), "video/mp4")
               if out else lip_key)
        with session_scope() as s:
            blk = s.query(Block).filter_by(job_id=ctx.job_id, index=idx).one()
            checkpoint.mark(blk, new_stage, key)
        shutil.rmtree(wd, ignore_errors=True)

    ctx.heartbeat()
    with session_scope() as s:
        job = load_job(s, ctx.job_id)
        finals = [b.final_key for b in job.blocks]
        audio_key, duration = job.audio_key, job.duration_s
    wd = ctx.sub("mux")
    with spend(ctx.job_id, "mux", costs.cpu_usd(costs.mux_cpu_s(duration)), cpu=True):
        paths = [s3.fetch(k, wd / f"b{i:04d}.mp4") for i, k in enumerate(finals)]
        audio = s3.fetch(audio_key, wd / "audio.wav") if audio_key else None
        out = mux(paths, audio, wd / "output.mp4", wd)
    key = s3.upload(out, s3.job_key(ctx.job_id, "output.mp4"), "video/mp4")
    with session_scope() as s:
        job = s.get(Job, ctx.job_id)
        job.output_key, job.state, job.error = key, JobState.done, None


def run(ctx: Ctx) -> None:
    prepare(ctx)
    analyze(ctx)
    build(ctx)
    render_all(ctx)
    audio_all(ctx)
    post_all(ctx)
