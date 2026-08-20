import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

points_selected = []


def pick_two_points(preliminary_image):
    global points_selected
    points_selected = []
    image = preliminary_image.copy()
    prompt_text = "Click to select 2 points, then press 'c'. Press 'r' to reset."
    cv2.putText(image, prompt_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points_selected) < 2:
                points_selected.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("image", image)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_event)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points_selected = []
            image = preliminary_image.copy()
            cv2.putText(image, prompt_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("image", image)
        elif key == ord("c") and len(points_selected) == 2:
            break

    cv2.destroyAllWindows()
    return points_selected


def extract_frames_from_video(video_path, output_folder, frame_frequency=1):
    """Extracts frames from a video file and saves them as sorted sequential JPGs."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error: OpenCV could not open video file: {video_path}")

    print(f"Extracting frames from video '{video_path}' into temporary workspace '{output_folder}'...")
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_frequency == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_idx:05d}.JPG")
            cv2.imwrite(frame_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"Successfully extracted {saved_count} frames from video.")
    return saved_count


def run_extensometer_analysis(
    input_path,
    output_csv_path,
    window_size=64,
    frame_frequency=1,
    points_file=None,
    start_frame=0,
    end_frame=None,
):
    """Executes extensometer analysis on a directory or video file by tracking 2 points."""
    is_video = input_path.lower().endswith((".mov", ".mp4", ".avi"))

    if is_video:
        folder_path = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        temp_folder = os.path.join(folder_path, f"{base_name}_temp_frames")

        need_extraction = True
        if os.path.exists(temp_folder):
            existing_frames = glob.glob(os.path.join(temp_folder, "*.JPG")) + glob.glob(
                os.path.join(temp_folder, "*.jpg")
            )
            if len(existing_frames) >= 2:
                need_extraction = False
                print(
                    f"Using {len(existing_frames)} existing frames found in "
                    f"'{temp_folder}'. Skipping extraction."
                )

        if need_extraction:
            num_frames = extract_frames_from_video(input_path, temp_folder, frame_frequency)
            if num_frames < 2:
                print("Error: Need at least two frames extracted to perform analysis.")
                return
        folder_path = temp_folder
    else:
        folder_path = input_path
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")

    search_pattern = os.path.join(folder_path, "*.JPG")
    img_list = sorted(glob.glob(search_pattern))
    if not img_list:
        search_pattern = os.path.join(folder_path, "*.jpg")
        img_list = sorted(glob.glob(search_pattern))

    if len(img_list) < 2:
        print(f"Error: Need at least two frames in {folder_path} to perform analysis.")
        return

    last_frame = len(img_list) - 1
    if end_frame is None:
        end_frame = last_frame
    if start_frame < 0 or end_frame > last_frame or start_frame > end_frame:
        print(
            f"Error: invalid frame range [{start_frame}, {end_frame}] for {len(img_list)} available frames "
            f"(valid range is [0, {last_frame}])."
        )
        return
    img_list = img_list[start_frame : end_frame + 1]
    if len(img_list) < 2:
        print(
            f"Error: Need at least two frames in the selected range "
            f"[{start_frame}, {end_frame}] to perform analysis."
        )
        return

    print("Initializing tracking...")
    img_ref_color = cv2.imread(img_list[0])
    img_ref = cv2.cvtColor(img_ref_color, cv2.COLOR_BGR2GRAY)

    pts = None
    if points_file and os.path.exists(points_file):
        try:
            with open(points_file, "r") as f:
                pts = json.load(f)
            print(f"Loaded previously selected points from {points_file}: {pts}")
        except Exception as e:
            print(f"Could not load points file: {e}")

    if not pts or len(pts) < 2:
        pts = pick_two_points(img_ref_color)
        print(f"Selected points: {pts[0]} and {pts[1]}")
        if points_file:
            try:
                with open(points_file, "w") as f:
                    json.dump(pts, f)
                print(f"Saved selected points to {points_file}")
            except Exception as e:
                print(f"Could not save points file: {e}")

    point_to_process = np.array(
        [[np.float32(pts[0][0]), np.float32(pts[0][1])], [np.float32(pts[1][0]), np.float32(pts[1][1])]]
    )

    lk_params = dict(
        winSize=(window_size, window_size),
        maxLevel=10,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    pt1_x, pt1_y = point_to_process[0].ravel()
    pt2_x, pt2_y = point_to_process[1].ravel()
    initial_distance = np.sqrt((pt2_x - pt1_x) ** 2 + (pt2_y - pt1_y) ** 2)
    print(f"Initial distance between points: {initial_distance:.4f} pixels")

    img_with_points = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_with_points, (int(pt1_x), int(pt1_y)), 10, (0, 0, 255), -1)
    cv2.circle(img_with_points, (int(pt2_x), int(pt2_y)), 10, (0, 0, 255), -1)
    cv2.line(img_with_points, (int(pt1_x), int(pt1_y)), (int(pt2_x), int(pt2_y)), (0, 255, 0), 2)
    img_filename = output_csv_path.replace(".csv", "_points_image.jpg")
    cv2.imwrite(img_filename, img_with_points)
    print(f"Saved tracking reference image to {img_filename}")

    results = []
    results.append(
        {
            "frame": os.path.basename(img_list[0]),
            "pt1_x": pt1_x,
            "pt1_y": pt1_y,
            "pt2_x": pt2_x,
            "pt2_y": pt2_y,
            "distance": initial_distance,
            "strain": 0.0,
        }
    )

    for i in range(len(img_list) - 1):
        if i % 10 == 0:
            print(f"Processing frame {i + 1} / {len(img_list) - 1}...")
        image_str = cv2.imread(img_list[i + 1], 0)

        final_point, st, err = cv2.calcOpticalFlowPyrLK(
            img_ref, image_str, point_to_process, None, **lk_params
        )

        pt1_x, pt1_y = final_point[0].ravel()
        pt2_x, pt2_y = final_point[1].ravel()

        distance = np.sqrt((pt2_x - pt1_x) ** 2 + (pt2_y - pt1_y) ** 2)
        strain = (distance - initial_distance) / initial_distance

        results.append(
            {
                "frame": os.path.basename(img_list[i + 1]),
                "pt1_x": pt1_x,
                "pt1_y": pt1_y,
                "pt2_x": pt2_x,
                "pt2_y": pt2_y,
                "distance": distance,
                "strain": strain,
            }
        )

        point_to_process = final_point
        img_ref = image_str

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Tracking complete. Saving results to '{output_csv_path}'...")
    with open(output_csv_path, "w", newline="") as csvfile:
        fieldnames = ["frame", "pt1_x", "pt1_y", "pt2_x", "pt2_y", "distance", "strain"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"SUCCESS: Strain history saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Extensometer: Track 2 points in a video and compute strain."
    )
    parser.add_argument(
        "input_source", type=str, help="Path to a video file (.mov/.mp4/.avi) or a directory of *.JPG frames"
    )
    parser.add_argument(
        "--window-size", type=int, default=64, help="Tracking window size in pixels (default: 64)"
    )
    parser.add_argument(
        "--frame-frequency",
        type=int,
        default=1,
        help="Extract every Nth frame if input is a video (default: 1)",
    )
    parser.add_argument(
        "--points-file",
        type=str,
        default=None,
        help="JSON file to save/load the 2 tracked points, to avoid repeated interaction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <input>_extensometer.csv next to the input)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame (index into the sorted/extracted frame list) to consider (default: 0)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame (index into the sorted/extracted frame list) to consider "
        "(default: last available frame)",
    )
    args = parser.parse_args()

    if args.output:
        output_data_file = args.output
    else:
        clean_path = args.input_source.rstrip(os.sep)
        if clean_path.lower().endswith((".mov", ".mp4", ".avi")):
            base_name = os.path.splitext(clean_path)[0]
        else:
            base_name = clean_path
        output_data_file = f"{base_name}_extensometer.csv"

    run_extensometer_analysis(
        args.input_source,
        output_data_file,
        window_size=args.window_size,
        frame_frequency=args.frame_frequency,
        points_file=args.points_file,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
