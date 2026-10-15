"""Run official Ultralytics inference for detect / segment / pose / track."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "ultralytics"
# Parent folder is also named ultralytics; keep it from shadowing the installed package.
sys.path = [p for p in sys.path if Path(p).resolve() != ROOT]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
from ultralytics import YOLO
from ultralytics.utils.downloads import download
DEMO = ROOT / "official_demo"
INPUTS = DEMO / "inputs"
RESULTS = DEMO / "results"
PREVIEWS = DEMO / "previews"

BUS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"
VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/decelera_portrait_min.mov"
CLIP_FRAMES = 90  # ~3s, enough to show IDs without a long wait


def clip_video(src: Path, dst: Path, max_frames: int = CLIP_FRAMES) -> Path:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    if n == 0:
        raise RuntimeError(f"No frames read from {src}")
    print(f"clipped {n} frames -> {dst}")
    return dst


def save_preview(image_path: Path, name: str) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read result image: {image_path}")
    out = PREVIEWS / f"{name}.jpg"
    cv2.imwrite(str(out), img)
    print(f"preview: {out}")


def summarize(task: str, results) -> None:
    r = results[0]
    boxes = 0 if r.boxes is None else len(r.boxes)
    masks = 0 if r.masks is None else len(r.masks)
    kpts = 0 if r.keypoints is None else len(r.keypoints)
    ids = []
    if r.boxes is not None and r.boxes.id is not None:
        ids = [int(x) for x in r.boxes.id.tolist()]
    print(f"[{task}] boxes={boxes} masks={masks} keypoints={kpts} ids={ids}")


def main() -> None:
    for p in (INPUTS, RESULTS, PREVIEWS):
        p.mkdir(parents=True, exist_ok=True)

    download(BUS_URL, dir=INPUTS)
    download(VIDEO_URL, dir=INPUTS)
    bus = INPUTS / "bus.jpg"
    raw_video = INPUTS / "decelera_portrait_min.mov"
    clip = clip_video(raw_video, INPUTS / "track_clip.mp4")

    common = dict(device=0, project=str(RESULTS), exist_ok=True, save=True, verbose=True)

    detect = YOLO("yolo26n.pt").predict(source=str(bus), name="detect", **common)
    summarize("detect", detect)
    save_preview(RESULTS / "detect" / "bus.jpg", "01_detect")

    segment = YOLO("yolo26n-seg.pt").predict(source=str(bus), name="segment", **common)
    summarize("segment", segment)
    save_preview(RESULTS / "segment" / "bus.jpg", "02_segment")

    pose = YOLO("yolo26n-pose.pt").predict(source=str(bus), name="pose", **common)
    summarize("pose", pose)
    save_preview(RESULTS / "pose" / "bus.jpg", "03_pose")

    track = YOLO("yolo26n.pt").track(
        source=str(clip),
        name="track",
        tracker="tracktrack.yaml",
        persist=True,
        **common,
    )
    last = track[-1]
    summarize("track-last-frame", [last])
    frame = last.plot()
    cv2.imwrite(str(PREVIEWS / "04_track.jpg"), frame)
    print(f"preview: {PREVIEWS / '04_track.jpg'}")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
