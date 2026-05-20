from pathlib import Path
import shutil
from PIL import Image
import os
import math
import ffmpeg

def upscale_video(
    input_video: Path,
    output_video: Path,
    scale_factor: int = 2,
    method: str = "lanczos"
):
    """
    Upscale une vidéo existante sans IA.

    method:
        - lanczos (recommandé)
        - bicubic
        - bilinear
    """

    input_video = Path(input_video)
    output_video = Path(output_video)

    print(f"🔎 Upscaling vidéo x{scale_factor} ({method})...")

    (
        ffmpeg
        .input(str(input_video))
        .filter("scale",
                f"iw*{scale_factor}",
                f"ih*{scale_factor}",
                flags=method)
        .output(
            str(output_video),
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=18  # qualité élevée
        )
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"✅ Vidéo upscalée générée : {output_video}")


# -------------------------
# Video save
# -------------------------

# -------------------------
# Video utilities
# -------------------------
def save_frames_as_video(frames, output_path, fps=12):
    temp_dir = Path("temp_frames")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")

    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)


# -------------------------
# Save video
# -------------------------
def save_frames_as_video_rmtmp(frames, output_path, fps=12):
    temp_dir = Path("temp_frames")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")
    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)
# -------------------------
# Video utilities
# -------------------------
def save_frames_as_video_ori(frames, output_path, fps=12):
    temp_dir = Path("temp_frames")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")

    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)
