import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO
import streamlit as st


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

                dist = np.hypot(
                    cx - cx_last,
                    cy - cy_last
                )

                if dist > self.max_distance:
                    continue

                dist_score = 1.0 - (
                    dist / self.max_distance
                )

                area_ratio = (
                    min(area, self.last_area)
                    /
                    max(area, self.last_area)
                )

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

            if best_det is not None:

                self.last_bbox = best_det[:4]

                self.last_center = (
                    (best_det[0] + best_det[2]) / 2,
                    (best_det[1] + best_det[3]) / 2
                )

                self.last_area = best_det[5]

                self.lost_frames = 0

                return self.last_bbox

            else:

                self.lost_frames += 1

                if self.lost_frames > self.max_lost:
                    self.state = "SEARCHING"

                return None


# ==========================================
# Processing
# ==========================================
def process_video(video_path):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
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
    # Load model
    # ==========================================
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
            f"Invalid video resolution: {width}x{height}"
        )

    if fps <= 0 or np.isnan(fps):

        fps = 25.0

    # ==========================================
    # IMPORTANT:
    # Write AVI first using MJPG.
    #
    # This is much more reliable with OpenCV
    # than directly writing MP4 with mp4v.
    # ==========================================
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

    tracker = SingleBallTracker()

    progress = st.progress(0)

    status = st.empty()

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
        # YOLO
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

            if w > 0 and h > 0:

                aspect_ratio = max(
                    w / h,
                    h / w
                )

            else:

                aspect_ratio = 99

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

            # Safety clamp
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

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

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
        # Write frame
        # ==========================================
        out.write(frame)

        # ==========================================
        # Progress
        # ==========================================
        if total_frames > 0:

            progress.progress(
                min(
                    frame_id / total_frames,
                    1.0
                )
            )

            status.text(
                f"Processing: "
                f"{frame_id}/{total_frames}"
            )

    # ==========================================
    # Close video
    # ==========================================
    cap.release()
    out.release()

    progress.empty()

    status.text(
        "Converting output video..."
    )

    # ==========================================
    # Convert AVI -> MP4
    #
    # OpenCV may not have a proper H264 encoder,
    # so use FFmpeg if available.
    # ==========================================
    ffmpeg_command = (
        f'ffmpeg -y '
        f'-i "{temp_output}" '
        f'-c:v libx264 '
        f'-preset fast '
        f'-crf 23 '
        f'-pix_fmt yuv420p '
        f'-movflags +faststart '
        f'"{final_output}"'
    )

    return_code = os.system(
        ffmpeg_command
    )

    # ==========================================
    # If FFmpeg conversion succeeded
    # ==========================================
    if (
        return_code == 0
        and
        os.path.exists(final_output)
        and
        os.path.getsize(final_output) > 0
    ):

        try:
            os.remove(
                temp_output
            )
        except:
            pass

        status.success(
            "Analysis Completed"
        )

        return final_output

    # ==========================================
    # FFmpeg unavailable:
    # use AVI as fallback
    # ==========================================
    else:

        status.warning(
            "MP4 conversion failed. "
            "Using AVI output instead."
        )

        return temp_output


# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(
    page_title="Ball Tracker",
    layout="wide"
)

st.title(
    "⚽ Football Ball Tracker"
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


if uploaded_video:

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(
        uploaded_video.read()
    )

    temp_video.close()

    original_video = temp_video.name

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

        if st.button(
            "🚀 Analyze Video",
            use_container_width=True
        ):

            try:

                st.session_state.output_video = (
                    process_video(
                        original_video
                    )
                )

                st.session_state.view_mode = (
                    "analysis"
                )

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Original",
                use_container_width=True
            ):

                st.session_state.view_mode = (
                    "original"
                )

        with col2:

            if st.button(
                "Analysis",
                use_container_width=True
            ):

                st.session_state.view_mode = (
                    "analysis"
                )

    # ==========================================
    # Video Display
    # ==========================================
    with right:

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

            # Read video bytes
            with open(
                st.session_state.output_video,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )

            st.video(
                video_bytes
            )

        else:

            st.subheader(
                "Original Video"
            )

            st.video(
                original_video
            )

