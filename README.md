# dogbone-videoextensometer

Track two points in a video of a tensile test (e.g. a 3D-printed dogbone specimen) and compute
engineering strain over time, using OpenCV's Lucas-Kanade optical flow tracker. Includes a
companion plotting script to visualize the tracked points and the resulting strain curve.

## Installation

```bash
pip install -r requirements.txt
```

## Usage: `videoextensometer.py`

Three subcommands, meant to be used in sequence:

### 1. `locate` — find roughly where the loading event happens in a long recording

```bash
python3 videoextensometer.py locate path/to/video.mov
```

Coarse whole-frame phase-correlation scan across the whole video; prints the frame range where
motion actually happens, so you don't have to scrub through the recording by hand. Suggests a
`--start-frame`/`--end-frame` window for `track`.

| Flag | Description |
|---|---|
| `--step N` | frame stride for the coarse scan (default: 60) |

### 2. `autopoints` — auto-detect the specimen's gauge region and suggest two track points

```bash
python3 videoextensometer.py autopoints path/to/video.mov --frame 0
```

Detects the specimen's parallel gauge region via image processing (specimen mask → width(x)
profile → the longest, narrowest, genuinely shoulder-flanked flat run) and suggests two `--pt1`/
`--pt2` points well inside it, individually per test — a fixed point pair reused across tests can
drift into the shoulder well before the ramp ends if it doesn't leave enough margin for how far a
given specimen happens to stretch.

| Flag | Description |
|---|---|
| `--frame N` | frame to analyze (default: 0, the first frame) |
| `--margin F` | fractional inward margin from each end of the detected gauge region (default: 0.25) |
| `--tol F` | relative width tolerance above the minimum to still count as the gauge region (default: 0.08) |
| `--preview FILE` | save an annotated preview image showing the detected region and suggested points |

### 3. `track` — track the two points and compute strain

```bash
python3 videoextensometer.py track path/to/video.mov \
    --pt1 200,355 --pt2 500,355 --start-frame 0 --end-frame 1200 --out video_extensometer.csv
```

Tracks the two points frame-by-frame with pyramidal Lucas-Kanade optical flow (propagated from the
previous frame's position, not re-initialized from the original points every frame), with a
forward-backward consistency check per frame to flag possible tracking drift. No physical
pixel-to-mm calibration is needed — strain is a ratio: `strain(t) = distance(t) / distance(start_frame) - 1`.

An isolated corrupted/unreadable frame carries the previous tracked position forward (logged, not
silent); if more than 1% of the requested range is unreadable, tracking aborts loudly instead of
producing a fabricated flat trace.

Output CSV columns: `frame, pt1_x, pt1_y, pt2_x, pt2_y, distance, strain, fb_error_px`.

| Flag | Description |
|---|---|
| `--pt1 X,Y` / `--pt2 X,Y` | pixel coordinates of the two points to track (required) |
| `--start-frame N` / `--end-frame N` | frame range to track (required) |
| `--out FILE` | output CSV path (required) |
| `--preview FILE` | save a reference-frame image with the two starting points marked |

### 4. Visualize results: `plot_video_extensometer.py`

```bash
python3 plot_video_extensometer.py path/to/video_extensometer.csv --video path/to/video.mp4
```

Produces a figure with 4 sampled frames (evenly spaced across the tracked range) showing the
tracked points and connecting line, plus a strain-vs-frame-number plot with the sampled frames
marked. Saved as `<csv>_overview.png` and also shown interactively.

**Not yet updated for `videoextensometer.py`'s CSV format**: this script was written against the
retired `video_extensometer.py`, whose `frame` column held extracted-JPG filenames
(`frame_index_from_name` regex-parses a string); `videoextensometer.py track`'s CSV instead has a
plain integer `frame` column and no extracted-frames folder, so `load_frame`'s frames-dir path is
now unreachable and `frame_index_from_name` will raise on an int input. Use `--video` to seek
frames directly and fix `frame_index_from_name` (or bypass it) before relying on this script.

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
python3 videoextensometer.py locate data/test1.MOV
python3 videoextensometer.py autopoints data/test1.MOV --preview data/test1_autopoints.jpg
python3 videoextensometer.py track data/test1.MOV \
    --pt1 214,355 --pt2 486,355 --start-frame 40 --end-frame 980 --out data/test1_extensometer.csv
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
