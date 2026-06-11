import subprocess
import os
import argparse

def extract_frames(video_path, output_dir, fps=2):
    """
    Extract frames from a video file using ffmpeg.
    
    Args:
        video_path: path to input video file
        output_dir: directory to save extracted frames
        fps: frames per second to extract (default 2 = 1 frame every 0.5 seconds)
    """
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "frame_%04d.jpeg")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",        # jpeg quality (2 = high quality)
        output_pattern,
        "-y"                # overwrite if exists
    ]

    print(f"Extracting frames from: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"FPS: {fps}")
    print("Running...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        frames = [f for f in os.listdir(output_dir) if f.endswith(".jpeg")]
        print(f"Done! Extracted {len(frames)} frames.")
        print(f"Frames saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="inputs/frames", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second to extract")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps)
    