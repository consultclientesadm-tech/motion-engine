from db.models import Block, BlockStage, Job
from workers.render import checkpoint


def test_plan_blocks_30s():
    blocks = checkpoint.plan_blocks(75, 30)
    assert blocks == [(0, 30), (30, 60), (60, 75)]
    assert checkpoint.plan_blocks(60, 30) == [(0, 30), (30, 60)]


def _job(stages):
    job = Job(video_url="x", budget_usd=10, daily_budget_usd=90, duration_s=30 * len(stages))
    for i, st in enumerate(stages):
        b = Block(index=i, start_s=i * 30, end_s=(i + 1) * 30, stage=st, has_people=True)
        b.render_key = f"r{i}" if checkpoint.rank(st) >= 1 else None
        b.lipsync_key = f"l{i}" if checkpoint.rank(st) >= 2 else None
        b.final_key = f"f{i}" if checkpoint.rank(st) >= 3 else None
        job.blocks.append(b)
    return job


def test_resume_skips_done_blocks():
    job = _job([BlockStage.upscaled, BlockStage.rendered, BlockStage.pending])
    assert [b.index for b in checkpoint.pending(job, BlockStage.rendered)] == [2]
    assert [b.index for b in checkpoint.pending(job, BlockStage.lipsynced)] == [1, 2]


def test_verify_demotes_missing_files():
    job = _job([BlockStage.upscaled, BlockStage.lipsynced, BlockStage.rendered])
    present = {"r0", "l0", "r1", "f0"}  # l1 sumiu, r2 sumiu
    fixed = checkpoint.verify(job, lambda k: k in present)
    assert fixed == [1, 2]
    assert [b.stage for b in job.blocks] == [BlockStage.upscaled, BlockStage.rendered, BlockStage.pending]


def test_skipped_block_counts_as_done():
    job = _job([BlockStage.upscaled, BlockStage.pending])
    job.blocks[1].stage, job.blocks[1].final_key = BlockStage.skipped, "f1"
    assert checkpoint.blocks_done(job) == 2
    assert checkpoint.pending(job, BlockStage.rendered) == []
