"""Two-point video extensometer for dogbone tensile-test recordings.

Tracks two user-specified material points across a frame range of a video
and computes the engineering strain history from their changing pixel
separation. No physical pixel-to-mm calibration is needed: strain is a
ratio, so

    strain(t) = distance(t) / distance(start_frame) - 1

Tracking uses pyramidal Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK),
propagated frame-to-frame from the previous frame's tracked position (not
re-initialized from the original points every frame), with a
forward-backward consistency check per frame to flag possible tracking
drift (reported, not silently corrected).

Three subcommands:

  locate      Coarse whole-frame phase-correlation scan across an entire
              video, to find roughly where the specimen motion (loading
              event) occurs -- i.e., where to point `track` at, without
              having to scrub through a possibly very long recording by
              hand. Prints the frame index where the cumulative
              horizontal shift (relative to frame 0) first departs from
              ~0 and where it plateaus again.

  autopoints  Detect the specimen's parallel gauge region in one frame via
              image processing (specimen mask -> width(x) profile -> the
              longest, narrowest, genuinely shoulder-flanked flat run --
              see find_gauge_region) and suggest two --pt1/--pt2 track
              points well inside it, individually per test -- rather than
              a hand-picked (x, y) pair reused across every test of a
              material, which can drift into the shoulder before the
              ramp ends if it doesn't leave enough margin for how far a
              given test happens to stretch (confirmed directly for both
              A75V25 and A100V0). Also saves a preview showing the
              detected gauge boundaries and suggested points.

  track       Track two points across [--start-frame, --end-frame],
              writing a CSV (frame, pt1_x, pt1_y, pt2_x, pt2_y, distance,
              strain) plus a PNG preview of the first frame with both
              points marked, for visual confirmation before trusting the
              pixel coordinates (whether hand-picked or from autopoints).

Usage:
    python videoextensometer.py locate VIDEO.MOV [--step 60]

    python videoextensometer.py autopoints VIDEO.MOV [--frame 0] \\
        [--margin 0.25] [--tol 0.08] [--preview preview.png]

    python videoextensometer.py track VIDEO.MOV --pt1 X,Y --pt2 X,Y \\
        --start-frame N --end-frame N --out out.csv [--preview preview.png]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LK_PARAMS = dict(
    winSize=(31, 31),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def parse_point(s):
    x, y = s.split(",")
    return float(x), float(y)


def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx}")
    return frame


def cmd_locate(args):
    cap = cv2.VideoCapture(str(args.video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{args.video}: {n_frames} frames @ {fps:.4f} fps ({n_frames / fps:.1f}s)")

    ref = cv2.cvtColor(read_frame(cap, 0), cv2.COLOR_BGR2GRAY).astype(np.float32)
    idxs = list(range(0, n_frames, args.step))
    shifts = []
    for i in idxs:
        gray = cv2.cvtColor(read_frame(cap, i), cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ref, gray)
        shifts.append(abs(dx))
    cap.release()

    shifts = np.array(shifts)
    # phase correlation occasionally glitches to a spuriously low OR high
    # reading on an isolated frame (texture-poor background, JPEG-ish
    # compression artifacts); a short rolling median suppresses these
    # before thresholding, without smearing the real (much longer) ramp
    # transition.
    smoothed = pd.Series(shifts).rolling(5, center=True, min_periods=1).median().to_numpy()
    baseline = np.median(smoothed[: max(1, len(smoothed) // 10)])
    plateau = np.median(smoothed[-max(1, len(smoothed) // 10) :])
    moved = np.abs(smoothed - baseline) > 0.4 * abs(plateau - baseline) + 1.0
    onset_i = int(moved.argmax()) if moved.any() else None
    end_i = len(moved) - 1 - int(moved[::-1].argmax()) if moved.any() else None

    print(f"baseline |dx|~{baseline:.2f}px, plateau |dx|~{plateau:.2f}px")
    if onset_i is not None and end_i is not None:
        onset_frame, end_frame = idxs[onset_i], idxs[end_i]
        print(f"motion onset ~frame {onset_frame} (t={onset_frame / fps:.2f}s)")
        print(f"motion plateau ~frame {end_frame} (t={end_frame / fps:.2f}s)")
        pad = args.step * 3
        print(
            f"suggested track window: --start-frame {max(0, onset_frame - pad)} "
            f"--end-frame {min(n_frames - 1, end_frame + pad)}"
        )
    else:
        print("no clear onset/plateau found -- inspect the raw scan below")
    for i, s in zip(idxs, shifts):
        print(f"  frame {i:7d}  t={i / fps:7.2f}s  |dx|={s:8.3f}")


def cmd_track(args):
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)

    p0 = np.array([args.pt1, args.pt2], dtype=np.float32).reshape(-1, 1, 2)
    # seek once, then read sequentially -- re-seeking (cap.set) every frame
    # forces a keyframe search + forward decode on compressed video and is
    # drastically slower than a plain sequential cap.read() loop.
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {args.start_frame}")
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    preview = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
    for x, y in p0.reshape(-1, 2):
        cv2.drawMarker(preview, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
        cv2.circle(preview, (int(x), int(y)), 8, (0, 255, 0), 2)
    if args.preview:
        cv2.imwrite(str(args.preview), preview)
        print(f"saved preview -> {args.preview}")

    d0 = float(np.hypot(*(p0[1, 0] - p0[0, 0])))
    rows = [
        {
            "frame": args.start_frame,
            "pt1_x": float(p0[0, 0, 0]), "pt1_y": float(p0[0, 0, 1]),
            "pt2_x": float(p0[1, 0, 0]), "pt2_y": float(p0[1, 0, 1]),
            "distance": d0, "strain": 0.0, "fb_error_px": 0.0,
        }
    ]

    pts = p0
    max_fb_error = 0.0
    n_unreadable = 0
    total_frames = args.end_frame - args.start_frame
    # A handful of isolated corrupted frames (e.g. a container-level decode
    # glitch -- confirmed possible via direct cv2 probing, distinct from a
    # real end-of-stream) shouldn't abort tracking for the whole recording,
    # so those are tolerated below by carrying the previous frame's tracked
    # position forward. But a video that is corrupted MUCH more broadly
    # (confirmed to happen: one recording had 98.6% of its frames fail with
    # "partial file" errors, which the carry-forward handling below would
    # otherwise silently turn into a flat, fabricated strain trace instead
    # of surfacing the real problem) must fail loudly instead -- so unreadable
    # frames are capped at this fraction of the requested range.
    MAX_UNREADABLE_FRACTION = 0.01
    for frame_idx in range(args.start_frame + 1, args.end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            n_unreadable += 1
            if n_unreadable > max(5, MAX_UNREADABLE_FRACTION * total_frames):
                raise RuntimeError(
                    f"aborting: {n_unreadable} unreadable frames so far (>{100 * MAX_UNREADABLE_FRACTION:.0f}% "
                    f"of the requested {total_frames}-frame range) -- this looks like broader file corruption, "
                    f"not an isolated glitch; re-check/re-transfer the source video rather than trusting "
                    f"a carried-forward trace"
                )
            print(f"  WARNING frame {frame_idx}: unreadable (corrupted frame), "
                  f"carrying previous tracked position forward")
            last = rows[-1]
            rows.append({**last, "frame": frame_idx, "fb_error_px": float("nan")})
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK_PARAMS)
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, pts_new, None, **LK_PARAMS)
        fb_error = float(np.max(np.hypot(*(pts_back - pts).reshape(-1, 2).T)))
        max_fb_error = max(max_fb_error, fb_error)
        if not (status.all() and status_back.all()):
            print(f"  WARNING frame {frame_idx}: optical flow lost a point (status={status.ravel()})")

        d = float(np.hypot(*(pts_new[1, 0] - pts_new[0, 0])))
        rows.append(
            {
                "frame": frame_idx,
                "pt1_x": float(pts_new[0, 0, 0]), "pt1_y": float(pts_new[0, 0, 1]),
                "pt2_x": float(pts_new[1, 0, 0]), "pt2_y": float(pts_new[1, 0, 1]),
                "distance": d, "strain": d / d0 - 1.0, "fb_error_px": fb_error,
            }
        )
        pts = pts_new
        prev_gray = gray
    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(
        f"tracked frames {args.start_frame}-{args.end_frame} ({len(df)} frames, fps={fps:.4f}), "
        f"max forward-backward error {max_fb_error:.3f}px -> {args.out}"
    )
    print(f"strain range: [{df['strain'].min():.4f}, {df['strain'].max():.4f}]")


def fill_small_holes(mask, max_area):
    """Fill enclosed (non-border-touching) dark regions inside the mask
    up to `max_area` pixels -- surface speckle/texture noise on the
    specimen punches holes of this scale straight through the naive
    Otsu mask, confirmed directly on more than one recording, badly
    enough that a fixed-size morphological close alone doesn't reliably
    clear it without also being large enough to erode real shoulder
    curvature. A large enclosed hole (e.g. a grip clamp's dark
    screw-head, also confirmed present in some recordings) is left
    alone -- that one is load-bearing for specimen_profile's
    solid-column rejection, not noise to be papered over."""
    h, w = mask.shape
    inv = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and area <= max_area:
            out[labels == i] = 255
    return out


def detect_specimen_mask(frame):
    """Binary mask of the specimen (assumed brighter than the background,
    true for every recording seen in this project so far): threshold on
    the blurred grayscale frame relative to the BACKGROUND's own
    brightness, sampled from the four frame corners (assumed background
    in every recording seen) -- global Otsu was tried first and rejected:
    confirmed directly that some recordings have a specimen whose neck is
    only modestly brighter than the background (dim exposure or a
    shadowed gauge section), and Otsu's single whole-frame-histogram
    threshold sets the cut too high for that region even though the
    shoulders are bright enough, corrupting the neck into a mess of holes
    a fixed-size morphological close/small-hole-fill can't cleanly
    recover (the resulting "hole" is comparable in size to a legitimate
    large occlusion like a grip clamp, so area alone can't tell them
    apart either). Small enclosed holes are then filled (surface
    speckle/texture noise -- see fill_small_holes), then opened to remove
    any remaining small noise blobs, then only the largest connected
    component is kept (the specimen spans/exceeds the frame width, so
    it's reliably the biggest blob)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = blur.shape
    c = 40
    corners = [blur[:c, :c], blur[:c, -c:], blur[-c:, :c], blur[-c:, -c:]]
    # median across the 4 corners, not a pooled mean/std -- one corner
    # catching a lighting reflection (confirmed to happen) otherwise
    # drags a pooled estimate high enough to swallow the specimen too
    bg_level = float(np.median([p.mean() for p in corners]))
    thresh_val = bg_level + 55
    mask = (blur > thresh_val).astype(np.uint8) * 255
    if gray[mask == 255].mean() < gray[mask == 0].mean():
        mask = cv2.bitwise_not(mask)
    mask = fill_small_holes(mask, max_area=3000)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


def specimen_profile(mask, min_solid_frac=0.9):
    """Per-column top/bottom row of the specimen mask (-1 where the
    column has no specimen pixels, or where the specimen region isn't
    at least `min_solid_frac` solid between top and bottom -- rejects
    columns crossing an occluding foreground object, e.g. the grip
    clamp's dark screw-head visible in some recordings, which would
    otherwise read as a false, much-narrower "gauge" cross-section)."""
    h, w = mask.shape
    top = np.full(w, -1, dtype=int)
    bottom = np.full(w, -1, dtype=int)
    for x in range(w):
        rows = np.nonzero(mask[:, x])[0]
        if len(rows) == 0:
            continue
        t, b = rows.min(), rows.max()
        solid_frac = mask[t:b + 1, x].mean() / 255
        if solid_frac >= min_solid_frac:
            top[x] = t
            bottom[x] = b
    return top, bottom


def bridge_false_runs(flags, max_gap):
    """Flip any False run of length <= max_gap back to True, provided
    it's flanked by True on both sides -- used below to stop a couple of
    noise-driven columns from fragmenting one real, otherwise-contiguous
    run (confirmed necessary: pixel-level jitter in the specimen mask's
    top/bottom edge is enough, on its own, to knock a handful of columns
    out of the flatness test in the middle of an obviously flat gauge
    region)."""
    flags = flags.copy()
    idx = np.nonzero(~flags)[0]
    if len(idx) == 0:
        return flags
    splits = np.nonzero(np.diff(idx) > 1)[0]
    bounds = [0, *(splits + 1), len(idx)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        run = idx[a:b]
        if len(run) <= max_gap and run[0] > 0 and run[-1] < len(flags) - 1:
            flags[run] = True
    return flags


def find_gauge_region(top, bottom, tol, min_run_frac=0.08, bridge_gap=20,
                       flat_bridge_gap=45, flatness_window=41):
    """The parallel gauge section is, by dogbone-specimen design, the
    longest LOCALLY FLAT run of the width(x) profile that's also the
    NARROWEST such run -- not simply "columns near the global minimum
    width", which a foreground object other than the specimen (e.g. a
    grip clamp's dark screw-head, confirmed to appear in some
    recordings) can fool by reading as an even-narrower sliver than the
    true gauge, despite not being part of any real, sustained plateau.
    Small gaps in column validity (a few pixels of surface-texture noise
    that survived masking) are bridged by linear interpolation first.
    Small gaps in the flatness test itself (pixel-level mask jitter, or a
    tiny surface defect -- confirmed to otherwise fragment an obviously
    flat gauge region into several short runs) are bridged separately,
    with a deliberately larger gap: unlike `bridge_gap`, which risks
    bridging clean across a real, large invalid stretch (e.g. between the
    true gauge and a spurious grip-clamp reading) if set too high,
    bridging the already-computed boolean `flat` array is safe to be more
    generous with, since it only ever merges runs that individually
    already passed the flatness test."""
    w = len(top)
    valid = (top >= 0) & (bottom >= 0)
    width = pd.Series(np.where(valid, (bottom - top).astype(float), np.nan))
    width_filled = width.interpolate(limit=bridge_gap, limit_area="inside")
    valid_filled = width_filled.notna().to_numpy()

    roll = width_filled.rolling(flatness_window, center=True, min_periods=flatness_window // 2)
    local_range = (roll.max() - roll.min()).to_numpy()
    local_median = roll.median().to_numpy()
    flat = valid_filled & (local_range <= tol * local_median)
    flat = bridge_false_runs(flat, flat_bridge_gap)

    idx = np.nonzero(flat)[0]
    if len(idx) == 0:
        raise RuntimeError("no flat (candidate gauge) region found -- check the specimen mask")
    splits = np.nonzero(np.diff(idx) > 1)[0]
    bounds = [0, *(splits + 1), len(idx)]
    runs = [idx[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    long_runs = [r for r in runs if len(r) >= min_run_frac * w] or [max(runs, key=len)]
    width_arr = width_filled.to_numpy()

    def widens_outward(r, search=100, factor=1.3):
        """A true gauge is a neck BETWEEN two widening shoulders -- unlike
        a flat run sitting on other hardware in frame (e.g. the grip
        clamp's cylindrical barrel, confirmed to otherwise read as a
        second, spuriously narrow "gauge"), which isn't flanked by
        genuinely wider material on both sides."""
        median_w = np.nanmedian(width_arr[r])

        def side_widens(vals):
            vals = vals[~np.isnan(vals)]
            return len(vals) == 0 or vals.max() >= factor * median_w

        left = width_arr[max(0, r[0] - search):r[0]]
        right = width_arr[r[-1] + 1:min(len(width_arr), r[-1] + 1 + search)]
        return side_widens(left) and side_widens(right)

    flanked_runs = [r for r in long_runs if widens_outward(r)] or long_runs
    best = min(flanked_runs, key=lambda r: np.nanmedian(width_arr[r]))
    return int(best[0]), int(best[-1])


def cmd_autopoints(args):
    """Detect the specimen's parallel gauge region in one frame and
    suggest two track points well inside it (a fractional margin in from
    each end), each vertically centered between the specimen's own
    top/bottom edge at that column -- rather than a hand-picked, fixed
    (x, y) reused across every test, which (confirmed directly, for
    A75V25 and A100V0) can drift into the shoulder well before the ramp
    ends if it doesn't leave enough margin for how far a given test
    happens to stretch."""
    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {args.frame}")
    cap.release()

    mask = detect_specimen_mask(frame)
    top, bottom = specimen_profile(mask)
    left, right = find_gauge_region(top, bottom, args.tol)
    span = right - left
    x1 = int(round(left + args.margin * span))
    x2 = int(round(right - args.margin * span))

    def vcenter(x):
        # nearest column (either direction) with a valid top/bottom, in
        # case x itself landed on a small bridged gap in the mask
        valid = np.nonzero((top >= 0) & (bottom >= 0))[0]
        x = valid[np.argmin(np.abs(valid - x))]
        return int(round((top[x] + bottom[x]) / 2))

    y1 = vcenter(x1)
    y2 = vcenter(x2)

    print(f"gauge region: x=[{left}, {right}] ({span}px wide)")
    print(f"suggested points: --pt1 {x1},{y1} --pt2 {x2},{y2}")

    if args.preview:
        vis = frame.copy()
        for x in range(left, right, 4):
            if top[x] >= 0:
                cv2.circle(vis, (x, int(top[x])), 1, (0, 255, 255), -1)
                cv2.circle(vis, (x, int(bottom[x])), 1, (0, 255, 255), -1)
        cv2.line(vis, (left, 0), (left, frame.shape[0]), (0, 255, 0), 1)
        cv2.line(vis, (right, 0), (right, frame.shape[0]), (0, 255, 0), 1)
        cv2.drawMarker(vis, (x1, y1), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(vis, (x2, y2), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imwrite(str(args.preview), vis)
        print(f"saved preview -> {args.preview}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_locate = sub.add_parser("locate", help="coarse-scan a video for the motion window")
    p_locate.add_argument("video", type=Path)
    p_locate.add_argument("--step", type=int, default=60, help="frame stride for the coarse scan")
    p_locate.set_defaults(func=cmd_locate)

    p_track = sub.add_parser("track", help="track two points across a frame range")
    p_track.add_argument("video", type=Path)
    p_track.add_argument("--pt1", type=parse_point, required=True, help="X,Y pixel coords")
    p_track.add_argument("--pt2", type=parse_point, required=True, help="X,Y pixel coords")
    p_track.add_argument("--start-frame", type=int, required=True)
    p_track.add_argument("--end-frame", type=int, required=True)
    p_track.add_argument("--out", type=Path, required=True)
    p_track.add_argument("--preview", type=Path, default=None)
    p_track.set_defaults(func=cmd_track)

    p_auto = sub.add_parser("autopoints", help="auto-detect the specimen's gauge region and suggest track points")
    p_auto.add_argument("video", type=Path)
    p_auto.add_argument("--frame", type=int, default=0, help="frame to analyze (default: first frame)")
    p_auto.add_argument("--margin", type=float, default=0.25,
                         help="fractional inward margin from each end of the detected gauge region")
    p_auto.add_argument("--tol", type=float, default=0.08,
                         help="relative width tolerance above the minimum to still count as the gauge region")
    p_auto.add_argument("--preview", type=Path, default=None)
    p_auto.set_defaults(func=cmd_autopoints)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
