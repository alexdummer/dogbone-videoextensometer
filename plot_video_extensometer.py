import argparse
import os
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def frame_index_from_name(frame_name):
    match = re.search(r"(\d+)", frame_name)
    if not match:
        raise ValueError(f"Could not parse a frame index from '{frame_name}'")
    return int(match.group(1))


def load_frame(row, video_path=None, frames_dir=None):
    """Loads a frame as RGB, preferring an extracted JPG and falling back to seeking the video."""
    frame_name = row['frame']
    if frames_dir:
        path = os.path.join(frames_dir, frame_name)
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if video_path:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_from_name(frame_name))
        ret, img = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise FileNotFoundError(f"Could not load frame '{frame_name}' from frames_dir or video.")


def draw_points(img, row):
    img = img.copy()
    pt1 = (int(row['pt1_x']), int(row['pt1_y']))
    pt2 = (int(row['pt2_x']), int(row['pt2_y']))
    cv2.circle(img, pt1, 10, (255, 0, 0), -1)
    cv2.circle(img, pt2, 10, (255, 0, 0), -1)
    cv2.line(img, pt1, pt2, (0, 255, 0), 3)
    return img


def pick_sample_indices(index_values, count=4):
    index_values = np.asarray(index_values)
    count = min(count, len(index_values))
    positions = np.linspace(0, len(index_values) - 1, count).round().astype(int)
    return sorted(set(index_values[positions].tolist()))


def plot_video_extensometer(csv_path, video_path=None, frames_dir=None, output_path=None,
                            start_frame=0, end_frame=None):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: CSV has no rows.")
        return

    last_frame = len(df) - 1
    if end_frame is None:
        end_frame = last_frame
    if start_frame < 0 or end_frame > last_frame or start_frame > end_frame:
        print(f"Error: invalid frame range [{start_frame}, {end_frame}] for {len(df)} rows "
              f"(valid range is [0, {last_frame}]).")
        return
    df = df.loc[start_frame:end_frame]

    if frames_dir is None:
        base_name = os.path.basename(csv_path).replace("_extensometer.csv", "")
        guess = os.path.join(os.path.dirname(csv_path), f"{base_name}_temp_frames")
        if os.path.isdir(guess):
            frames_dir = guess

    if video_path is None and frames_dir is None:
        print("Error: could not locate frame images. Pass --video or --frames-dir.")
        return

    sample_indices = pick_sample_indices(df.index, 4)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, len(sample_indices), height_ratios=[1.2, 1])

    for col, idx in enumerate(sample_indices):
        row = df.loc[idx]
        img = draw_points(load_frame(row, video_path=video_path, frames_dir=frames_dir), row)
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Frame {idx}\nstrain={row['strain'] * 100.0:.2f}%")

    ax_strain = fig.add_subplot(gs[1, :])
    ax_strain.plot(df.index, df['strain'] * 100.0, marker='o', linestyle='-', markersize=3, color='blue')
    for idx in sample_indices:
        ax_strain.axvline(idx, color='red', linestyle='--', alpha=0.5)
    ax_strain.set_xlabel('Frame Number')
    ax_strain.set_ylabel('Strain (%)')
    ax_strain.set_title('Strain vs. Frame')
    ax_strain.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f"Video Extensometer: {os.path.basename(csv_path)}")
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.splitext(csv_path)[0] + "_overview.png"
    plt.savefig(output_path, dpi=200)
    print(f"SUCCESS: Plot saved as '{output_path}'.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 4 sampled frames with tracked points alongside the strain-vs-frame curve, "
                    "from a CSV produced by video_extensometer.py."
    )
    parser.add_argument("csv_file", type=str, help="Path to the *_extensometer.csv produced by video_extensometer.py")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to the original video, used to grab frames if the temp frames folder was removed")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Path to the folder of extracted frame JPGs (default: <video>_temp_frames next to the CSV)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: <csv>_overview.png)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame (row index in the CSV) to consider (default: 0)")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Last frame (row index in the CSV) to consider (default: last row)")
    args = parser.parse_args()

    plot_video_extensometer(args.csv_file, video_path=args.video, frames_dir=args.frames_dir,
                            output_path=args.output, start_frame=args.start_frame, end_frame=args.end_frame)
