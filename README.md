# dogbone-videoextensometer

Track two points in a video of a tensile test (e.g. a 3D-printed dogbone specimen) and compute
engineering strain over time, using OpenCV's Lucas-Kanade optical flow tracker. Includes a
companion plotting script to visualize the tracked points and the resulting strain curve.

## Installation

```bash
pip install -r requirements.txt
```

Requires a display (X11/Wayland) for the interactive point-selection window in
`video_extensometer.py`, unless you reuse a previously saved `--points-file`.

## Usage

### 1. Track points and compute strain: `video_extensometer.py`

```bash
python3 video_extensometer.py path/to/video.mp4
```

On first run this opens the first frame in a window — click 2 points to track, press `c` to
confirm (or `r` to reset the selection). It then tracks those points frame-by-frame and writes a
CSV of positions, distance, and engineering strain.

Input can be a video file (`.mov`, `.mp4`, `.avi`) or a directory of already-extracted `*.JPG`
frames.

Output:
- `<video>_extensometer.csv` — one row per frame: `frame, pt1_x, pt1_y, pt2_x, pt2_y, distance, strain`
- `<video>_extensometer_points_image.jpg` — reference frame with the selected points marked
- `<video>_temp_frames/` — extracted frames (reused on subsequent runs if present)

Options:

| Flag | Description |
|---|---|
| `--window-size N` | Lucas-Kanade tracking window size in pixels (default: 64) |
| `--frame-frequency N` | Extract every Nth frame from the video (default: 1, i.e. every frame) |
| `--points-file FILE` | JSON file to save/load the 2 selected points, to skip the interactive picker on repeat runs |
| `--output FILE` | Output CSV path (default: `<video>_extensometer.csv`) |
| `--start-frame N` | First frame index (into the sorted/extracted frame list) to track (default: 0) |
| `--end-frame N` | Last frame index to track (default: last available frame) |

### 2. Visualize results: `plot_video_extensometer.py`

```bash
python3 plot_video_extensometer.py path/to/video_extensometer.csv --video path/to/video.mp4
```

Produces a figure with 4 sampled frames (evenly spaced across the tracked range) showing the
tracked points and connecting line, plus a strain-vs-frame-number plot with the sampled frames
marked. Saved as `<csv>_overview.png` and also shown interactively.

Frames are read from the `<video>_temp_frames/` folder left behind by `video_extensometer.py` if
still present (auto-detected), otherwise pass `--video` to seek the exact frames from the original
video directly, or `--frames-dir` to point at a specific folder of extracted frames.

Options:

| Flag | Description |
|---|---|
| `--video FILE` | Original video, used to grab frames if the temp frames folder was removed |
| `--frames-dir DIR` | Folder of extracted frame JPGs (default: `<video>_temp_frames` next to the CSV) |
| `--output FILE` | Output PNG path (default: `<csv>_overview.png`) |
| `--start-frame N` | First frame (row index in the CSV) to include (default: 0) |
| `--end-frame N` | Last frame (row index in the CSV) to include (default: last row) |

## Example workflow

```bash
python3 video_extensometer.py data/test1.MOV --points-file data/test1_points.json
python3 plot_video_extensometer.py data/test1_extensometer.csv --video data/test1.MOV
```

## Development

This repo uses [pre-commit](https://pre-commit.com/) to run `black`, `isort`, and `flake8` (plus a
few basic hygiene checks) on every commit.

```bash
pip install pre-commit
pre-commit install
```

To run the checks manually against all files:

```bash
pre-commit run --all-files
```

## License

MIT — see [LICENSE](LICENSE).
