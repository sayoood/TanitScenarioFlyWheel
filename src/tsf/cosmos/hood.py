"""The ego's hood in the front wide camera (T1 closed loop).

Sim-Auto draws its rig's hood across the bottom of the front frames, with the same outline in the first chunk of all 14
T1 runs (480p and 720p), but the WSM controls leave that area black, so the hood is only as steady as the I2V chain that
carries it. A chunk generated without a conditioning frame draws its own: in the 40 s protected-left rerun the redraw at
29.3 s gave a light rounded hood and the one at 33.6 s none at all, and from there to the end the front view read like a
side camera. Along the chain the hood blurs as well (the correlation of its fine detail with chunk 0's fell from 0.88 to
0.37 over the first 18 chunks). :class:`Hood` keeps chunk 0's hood and pastes it, with a soft edge, over the front
frames of every later chunk: its videos, and its last frame, which the policy sees and the next chunk is conditioned
on, so the chain carries chunk 0's hood again after a redraw. A redraw that puts its own hood higher than chunk 0's
(the light one at 29.3 s) still shows that part above the pasted one.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HOOD_CAMS = ("camera_front_wide_120fov",)
# the hood, the wipers and the dark band below them down to the image's edge (the hood alone floated over the road on a
# redraw that drew none), as fractions of the width and height; measured at 1280x720 on the first chunk of 14 runs
OUTLINE = np.array([(165, 648), (350, 572), (910, 572), (1080, 648), (1095, 720), (150, 720)]) / (1280, 720)
# below the wipers the band is the road seen through the dashboard's reflection: chunk 0's markings pasted there stood
# still, so the band gets the reflection's median colour at half weight over the current road
BAND, BAND_WEIGHT = 664 / 720, 0.5


def _outline(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros(shape, np.float32)
    cv2.fillPoly(m, [np.round(OUTLINE * (w, h)).astype(np.int32)], 1.0)
    return m


def hood_alpha(shape: tuple[int, int], feather: float = 3.0) -> np.ndarray:
    """Weight of the hood at each pixel of an image of ``shape`` (h, w): 1 inside :data:`OUTLINE` above the band,
    ``BAND_WEIGHT`` in it, a soft edge of ``feather`` pixels (at 720 rows), shape (h, w, 1)."""
    m = _outline(shape)
    m[int(BAND * shape[0]):] *= BAND_WEIGHT
    return cv2.GaussianBlur(m, (0, 0), feather * shape[0] / 720)[..., None]


class Hood:
    """``reference`` takes chunk 0's last frames; calling it pastes their hood into a later chunk's front frames."""

    def __init__(self, cams: tuple[str, ...] = HOOD_CAMS):
        self.cams = cams
        self.ref: dict[str, np.ndarray] = {}

    def reference(self, frames: dict[str, str | Path]) -> None:
        """Chunk 0's last frames, the band below the wipers filled with its median colour."""
        self.ref = {}
        for c, p in frames.items():
            if c in self.cams:
                img = cv2.imread(str(p))
                band = _outline(img.shape[:2]) > 0
                band[:int(BAND * img.shape[0])] = False
                img[band] = np.median(img[band], 0)
                self.ref[c] = img

    def paste(self, img: np.ndarray, cam: str) -> np.ndarray:
        ref = self.ref.get(cam)
        if ref is None:
            return img
        if ref.shape != img.shape:
            ref = cv2.resize(ref, img.shape[1::-1], interpolation=cv2.INTER_AREA)
        a = hood_alpha(img.shape[:2])
        return np.round(a * ref + (1 - a) * img).astype(np.uint8)

    def __call__(self, last_frames: dict[str, str], videos: dict[str, str], fps: float = 30.0) -> None:
        """Paste the hood into the chunk's last frames (PNG) and every frame of its videos (re-encoded), in place."""
        import imageio.v2 as imageio
        for c in self.ref:
            if c in last_frames:
                p = str(last_frames[c])
                cv2.imwrite(p, self.paste(cv2.imread(p), c))
            if c in videos:
                src, tmp = Path(videos[c]), Path(videos[c]).with_suffix(".hood.mp4")
                cap = cv2.VideoCapture(str(src))
                wr = imageio.get_writer(str(tmp), fps=fps, codec="libx264", macro_block_size=1,
                                        output_params=["-crf", "16", "-pix_fmt", "yuv420p"])
                try:
                    while True:
                        ok, fr = cap.read()
                        if not ok:
                            break
                        wr.append_data(cv2.cvtColor(self.paste(fr, c), cv2.COLOR_BGR2RGB))
                finally:
                    wr.close()
                    cap.release()
                tmp.replace(src)
