import cv2
import numpy as np
import os
import tempfile
import subprocess
from datetime import datetime

from ultralytics import YOLO
import streamlit as st
import imageio_ffmpeg


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/best.pt"

CONF_THRESHOLD_SEARCH = 0.35
CONF_THRESHOLD_TRACK = 0.08

MIN_AREA = 10
MAX_AREA = 2000
MAX_ASPECT_RATIO = 1.8


# ==========================================
# Single Ball Tracker
# ==========================================

class SingleBallTracker:

    def __init__(self):

        self.state = "SEARCHING"

        self.last_bbox = None
        self.last_center = None
        self.last_area = None

        self.w_conf = 1.0
        self.w_dist = 2.5
        self.w_area = 1.5

        self.max_distance = 150
        self.lost_frames = 0
        self.max_lost = 5

    def process_frame(self, detections):

        # ==========================================
        # No detections
        # ==========================================

        if len(detections) == 0:

            self.lost_frames += 1

            if self.lost_frames > self.max_lost:
                self.state = "SEARCHING"

            return None

        best_det = None

        # ==========================================
        # SEARCHING
        # ==========================================

        if self.state == "SEARCHING":

            best_conf = CONF_THRESHOLD_SEARCH

            for det in detections:

                if det[4] > best_conf:

                    best_conf = det[4]
                    best_det = det

            if best_det is not None:

                self.state = "TRACKING"

                self.last_bbox = best_det[:4]

                self.last_center = (
                    (best_det[0] + best_det[2]) / 2,
                    (best_det[1] + best_det[3]) / 2
                )

                self.last_area = best_det[5]

                self.lost_frames = 0

                return self.last_bbox

            return None

        # ==========================================
        # TRACKING
        # ==========================================

        elif self.state == "TRACKING":

            best_score = -999

            cx_last, cy_last = self.last_center

            for det in detections:

                x1, y1, x2, y2, conf, area = det

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Distance from previous ball position
                dist = np.hypot(
                    cx - cx_last,
                    cy - cy_last
                )

                if dist > self.max_distance:
                    continue

                dist_score = 1.0 - (
                    dist / self.max_distance
                )

                # Area similarity
                if self.last_area > 0 and area > 0:

                    area_ratio = (
                        min(area, self.last_area)
                        /
                        max(area, self.last_area)
                    )

                else:

                    area_ratio = 0

                # Combined score
                score = (
                    self.w_conf * conf
                    +
                    self.w_dist * dist_score
                    +
                    self.w_area * area_ratio
                )

                if score > best_score:

                    best_score = score
                    best_det = det

            # ==========================================
            # Valid tracking detection
            # ==========================================

            if best_det is not None:

                self.last_bbox = best_det[:4]

                self.last_center = (
                    (best_det[0] + best_det[2]) / 2,
                    (best_det[1] + best_det[3]) / 2
                )

                self.last_area = best_det[5]

                self.lost_frames = 0

                return self.last_bbox

            # ==========================================
            # Ball temporarily lost
            # ==========================================

            else:

                self.lost_frames += 1

                if self.lost_frames > self.max_lost:
                    self.state = "SEARCHING"

                return None


# ==========================================
# Processing Video
# ==========================================

def process_video(video_path):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = os.path.join(
        "runs",
        "detect",
        f"exp_{timestamp}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ==========================================
    # Output files
    # ==========================================

    temp_output = os.path.join(
        output_dir,
        "output_temp.avi"
    )

    final_output = os.path.join(
        output_dir,
        "output.mp4"
    )

    # ==========================================
    # Load YOLO model
    # ==========================================

    status = st.empty()

    status.info(
        "Loading YOLO model..."
    )

    model = YOLO(MODEL_PATH)

    # ==========================================
    # Open input video
    # ==========================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open input video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # ==========================================
    # Safety checks
    # ==========================================

    if width <= 0 or height <= 0:

        cap.release()

        raise RuntimeError(
            f"Invalid video resolution: "
            f"{width}x{height}"
        )

    if fps <= 0 or np.isnan(fps):

        fps = 25.0

    # ==========================================
    # Create temporary AVI
    #
    # MJPG is usually reliable with OpenCV.
    # ==========================================

    status.info(
        "Preparing video output..."
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    out = cv2.VideoWriter(
        temp_output,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create temporary output video."
        )

    # ==========================================
    # Initialize tracker
    # ==========================================

    tracker = SingleBallTracker()

    progress = st.progress(0)

    frame_id = 0

    # ==========================================
    # Processing loop
    # ==========================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        # ==========================================
        # YOLO Prediction
        # ==========================================

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD_TRACK,
            verbose=False
        )

        candidates = []

        # ==========================================
        # Candidate filtering
        # ==========================================

        if (
            results
            and
            len(results) > 0
            and
            results[0].boxes is not None
        ):

            for box in results[0].boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                conf = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )

                w = x2 - x1
                h = y2 - y1

                area = w * h

                # ==========================================
                # Valid dimensions
                # ==========================================

                if w > 0 and h > 0:

                    aspect_ratio = max(
                        w / h,
                        h / w
                    )

                else:

                    aspect_ratio = 99

                # ==========================================
                # Geometry filtering
                # ==========================================

                if (
                    MIN_AREA < area < MAX_AREA
                    and
                    aspect_ratio < MAX_ASPECT_RATIO
                ):

                    candidates.append(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                            conf,
                            area
                        ]
                    )

        # ==========================================
        # Tracker
        # ==========================================

        ball_bbox = tracker.process_frame(
            candidates
        )

        # ==========================================
        # Draw Ball
        # ==========================================

        if ball_bbox is not None:

            x1, y1, x2, y2 = map(
                int,
                ball_bbox
            )

            # ==========================================
            # Safety clamp
            # ==========================================

            x1 = max(
                0,
                min(x1, width - 1)
            )

            x2 = max(
                0,
                min(x2, width - 1)
            )

            y1 = max(
                0,
                min(y1, height - 1)
            )

            y2 = max(
                0,
                min(y2, height - 1)
            )

            # ==========================================
            # Ball bounding box
            # ==========================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==========================================
            # Ball center
            # ==========================================

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )

        # ==========================================
        # Write processed frame
        # ==========================================

        out.write(frame)

        # ==========================================
        # Progress
        # ==========================================

        if total_frames > 0:

            progress_value = (
                frame_id / total_frames
            )

            progress.progress(
                min(progress_value, 1.0)
            )

            status.text(
                f"Processing video: "
                f"{frame_id}/{total_frames}"
            )

    # ==========================================
    # Close video
    # ==========================================

    cap.release()
    out.release()

    progress.empty()

    # ==========================================
    # Verify temporary AVI
    # ==========================================

    if (
        not os.path.exists(temp_output)
        or
        os.path.getsize(temp_output) == 0
    ):

        raise RuntimeError(
            "Temporary AVI video was not created correctly."
        )

    # ==========================================
    # Convert AVI -> MP4
    #
    # IMPORTANT:
    # We use imageio-ffmpeg so the deployment
    # does NOT need a system FFmpeg installation.
    # ==========================================

    status.info(
        "Converting output video to MP4..."
    )

    try:

        # Get FFmpeg executable bundled with
        # imageio-ffmpeg
        ffmpeg_exe = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        # ==========================================
        # FFmpeg command
        # ==========================================

        command = [
            ffmpeg_exe,

            "-y",

            "-i",
            temp_output,

            # H.264 video
            "-c:v",
            "libx264",

            # Encoding speed
            "-preset",
            "fast",

            # Quality
            "-crf",
            "23",

            # Browser-compatible pixel format
            "-pix_fmt",
            "yuv420p",

            # Better streaming/browser compatibility
            "-movflags",
            "+faststart",

            final_output
        ]

        # ==========================================
        # Run FFmpeg
        # ==========================================

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # ==========================================
        # Successful conversion
        # ==========================================

        if (
            result.returncode == 0
            and
            os.path.exists(final_output)
            and
            os.path.getsize(final_output) > 0
        ):

            # Remove temporary AVI
            try:

                os.remove(
                    temp_output
                )

            except Exception:
                pass

            status.success(
                "Analysis Completed Successfully!"
            )

            return final_output

        # ==========================================
        # FFmpeg conversion failed
        # ==========================================

        else:

            error_message = result.stderr

            if not error_message:

                error_message = (
                    "Unknown FFmpeg conversion error."
                )

            st.error(
                "MP4 conversion failed."
            )

            st.code(
                error_message[-3000:]
            )

            # Return AVI so the user still
            # gets the processed video
            return temp_output

    # ==========================================
    # FFmpeg exception
    # ==========================================

    except Exception as e:

        st.error(
            f"FFmpeg error: {e}"
        )

        # AVI fallback
        return temp_output


# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(
    page_title="Football Ball Tracker",
    page_icon="⚽",
    layout="wide"
)

st.title(
    "⚽ Football Ball Tracker"
)

st.caption(
    "YOLO Ball Detection + Single Ball Tracking"
)


# ==========================================
# Session State
# ==========================================

if "output_video" not in st.session_state:

    st.session_state.output_video = None


if "view_mode" not in st.session_state:

    st.session_state.view_mode = "original"


# ==========================================
# Upload
# ==========================================

uploaded_video = st.file_uploader(
    "Upload Match Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# ==========================================
# Video Uploaded
# ==========================================

if uploaded_video:

    # ==========================================
    # Save uploaded video temporarily
    # ==========================================

    file_extension = os.path.splitext(
        uploaded_video.name
    )[1].lower()

    if not file_extension:

        file_extension = ".mp4"

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    )

    temp_video.write(
        uploaded_video.read()
    )

    temp_video.close()

    original_video = temp_video.name

    # ==========================================
    # Layout
    # ==========================================

    left, right = st.columns(
        [1, 2]
    )

    # ==========================================
    # Controls
    # ==========================================

    with left:

        st.subheader(
            "Controls"
        )

        # ==========================================
        # Analyze
        # ==========================================

        if st.button(
            "🚀 Analyze Video",
            use_container_width=True
        ):

            try:

                # Reset previous output
                st.session_state.output_video = None

                # Process
                output_path = process_video(
                    original_video
                )

                # Save result
                st.session_state.output_video = (
                    output_path
                )

                st.session_state.view_mode = (
                    "analysis"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Error during analysis: {e}"
                )

        st.markdown("---")

        # ==========================================
        # View buttons
        # ==========================================

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Original",
                use_container_width=True
            ):

                st.session_state.view_mode = (
                    "original"
                )

                st.rerun()

        with col2:

            if st.button(
                "Analysis",
                use_container_width=True
            ):

                st.session_state.view_mode = (
                    "analysis"
                )

                st.rerun()

    # ==========================================
    # Video Display
    # ==========================================

    with right:

        # ==========================================
        # Analysis video
        # ==========================================

        if (
            st.session_state.view_mode
            == "analysis"
            and
            st.session_state.output_video
            and
            os.path.exists(
                st.session_state.output_video
            )
        ):

            st.subheader(
                "Analysis Result"
            )

            output_path = (
                st.session_state.output_video
            )

            # ==========================================
            # MP4
            # ==========================================

            if output_path.lower().endswith(
                ".mp4"
            ):

                st.video(
                    output_path
                )

            # ==========================================
            # AVI fallback
            # ==========================================

            else:

                st.warning(
                    "The processed video is AVI "
                    "because MP4 conversion failed."
                )

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )

                st.download_button(
                    label="⬇️ Download AVI Result",
                    data=video_bytes,
                    file_name="output.avi",
                    mime="video/x-msvideo",
                    use_container_width=True
                )

        # ==========================================
        # Original video
        # ==========================================

        else:

            st.subheader(
                "Original Video"
            )

            st.video(
                original_video
            )