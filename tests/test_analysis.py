import json
import math

import numpy as np

from workers.analyze.assign import TrackSummary, auto_assign
from workers.analyze.blockdata import BlockAnalysis
from workers.retarget.retarget import BONES, REST_ANGLES, image_angle, local_angles, retarget_block


def test_blockdata_roundtrip_and_presence():
    k = np.zeros((133, 3), np.float32)
    rows = [(10, 1, np.array([0, 0, 10, 10]), k), (10, 2, np.array([5, 5, 9, 9]), k),
            (12, 1, np.array([1, 1, 11, 11]), k)]
    ba = BlockAnalysis.from_rows(rows, 10, 20)
    ba2 = BlockAnalysis.from_bytes(ba.to_bytes())
    assert ba2.max_people() == 2
    assert ba2.presence_ratio() == 0.2
    assert ba2.tracks() == [1, 2]
    assert list(ba2.last_boxes()[1]) == [1, 1, 11, 11]


def test_image_angle_convention():
    o = np.array([0.0, 0.0])
    assert image_angle(o, np.array([0.0, -1.0])) == 0          # para cima
    assert image_angle(o, np.array([1.0, 0.0])) == 90          # para a direita (+X)
    assert abs(image_angle(o, np.array([0.0, 1.0]))) == 180    # para baixo


def test_rest_pose_gives_zero_local_rotation():
    assert all(abs(v) < 1e-9 for v in local_angles(dict(REST_ANGLES)).values())


def _pose(arm_l_angle):
    """Pessoa de frente: quadril em (100,200), ombros em y=110."""
    k = np.zeros((133, 3), np.float32)
    k[:, 2] = 1.0
    pts = {0: (100, 80), 5: (110, 110), 6: (90, 110), 11: (105, 200), 12: (95, 200),
           13: (105, 245), 14: (95, 245), 15: (105, 290), 16: (95, 290), 8: (80, 140), 10: (75, 170)}
    a = math.radians(arm_l_angle)
    pts[7] = (110 + 30 * math.sin(a), 110 - 30 * math.cos(a))
    pts[9] = (pts[7][0] + 25 * math.sin(a), pts[7][1] - 25 * math.cos(a))
    for i, (x, y) in pts.items():
        k[i, :2] = (x, y)
    return k


def test_retarget_block_follows_arm():
    rows = [(0, 7, np.array([60, 60, 140, 300]), _pose(90))]
    out = retarget_block(BlockAnalysis.from_rows(rows, 0, 1), {7: "Ana"}, 200, 300)
    actor = out["frames"][0]["actors"][0]
    assert actor["character"] == "Ana"
    assert abs(actor["angles"]["upper_arm.L"] - 90) < 1e-3     # braço esticado na horizontal
    assert abs(actor["angles"]["thigh.L"] - 180) < 1e-3
    assert set(out["bones"]) == set(BONES)
    assert actor["face"] is not None and 0 < actor["root"][0] < 1
    json.dumps(out)  # vai para o Blender como JSON: nada de np.float32


def test_untracked_people_are_not_rendered():
    rows = [(0, 9, np.array([0, 0, 10, 10]), _pose(0))]
    out = retarget_block(BlockAnalysis.from_rows(rows, 0, 1), {7: "Ana"}, 200, 300)
    assert out["frames"][0]["actors"] == []


def T(i, first, last, cx_first, cx_last=None):
    cx_last = cx_first if cx_last is None else cx_last
    return TrackSummary(i, first, last, last - first + 1, [cx_first - 5, 0, cx_first + 5, 10],
                        [cx_last - 5, 0, cx_last + 5, 10])


def test_auto_assign_left_to_right_and_reconnect():
    tracks = [T(1, 0, 100, 800), T(2, 0, 100, 100), T(3, 5, 100, 450),
              T(4, 120, 300, 790),     # pessoa da direita reaparece com ID novo
              T(5, 0, 10, 300)]        # ruído curto
    m = auto_assign(tracks, ["A", "B", "C"], frame_w=900, min_frames=30)
    assert m == {"A": [2], "B": [3], "C": [1, 4]}
