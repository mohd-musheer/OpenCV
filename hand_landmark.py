import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# MediaPipe Hand Landmarker
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# MediaPipe Configuration
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


# ============================================================
# Canvas
# ============================================================

canvas = None

previous_point = None
smooth_point = None


# ============================================================
# Timestamp
# ============================================================

timestamp_ms = 0


# ============================================================
# Colors
#
# OpenCV uses BGR
#
# These colors are intentionally strong so they remain
# visible over white, pink, and camera backgrounds.
# ============================================================

COLOR_WRIST = (40, 40, 40)          # Dark gray

COLOR_THUMB = (255, 80, 40)         # Blue / Orange
COLOR_INDEX = (255, 0, 255)         # Magenta
COLOR_MIDDLE = (0, 200, 255)        # Yellow
COLOR_RING = (0, 180, 0)            # Green
COLOR_PINKY = (180, 80, 255)        # Purple

COLOR_PALM = (255, 255, 255)        # White
COLOR_POINT_OUTLINE = (30, 30, 30)  # Dark outline


# ============================================================
# Hand Skeleton Connections
#
# MediaPipe Hand landmarks:
#
# 0  = Wrist
#
# Thumb:
# 1 -> 2 -> 3 -> 4
#
# Index:
# 5 -> 6 -> 7 -> 8
#
# Middle:
# 9 -> 10 -> 11 -> 12
#
# Ring:
# 13 -> 14 -> 15 -> 16
#
# Pinky:
# 17 -> 18 -> 19 -> 20
#
# Palm connections:
# 0 -> 1
# 0 -> 5
# 0 -> 9
# 0 -> 13
# 0 -> 17
#
# Finger base connections:
# 5 -> 9
# 9 -> 13
# 13 -> 17
# ============================================================

HAND_CONNECTIONS = [
    # ---------------------------
    # Thumb
    # ---------------------------
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # ---------------------------
    # Index
    # ---------------------------
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # ---------------------------
    # Middle
    # ---------------------------
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # ---------------------------
    # Ring
    # ---------------------------
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # ---------------------------
    # Pinky
    # ---------------------------
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # ---------------------------
    # Palm
    # ---------------------------
    (5, 9),
    (9, 13),
    (13, 17),
]


# ============================================================
# Function: Distance
# ============================================================

def distance(point1, point2):

    return np.linalg.norm(
        np.array(point1, dtype=np.float32)
        -
        np.array(point2, dtype=np.float32)
    )


# ============================================================
# Function: Draw Hand Skeleton
# ============================================================

def draw_hand_landmarks(frame, points):

    # ========================================================
    # Draw Connections
    # ========================================================

    for connection in HAND_CONNECTIONS:

        start_index, end_index = connection

        start_point = points[start_index]
        end_point = points[end_index]


        # ---------------------------------------------
        # Decide color based on finger
        # ---------------------------------------------

        if start_index in [0, 1, 2, 3, 4] and \
           end_index in [0, 1, 2, 3, 4]:

            color = COLOR_THUMB

        elif start_index in [5, 6, 7, 8] and \
             end_index in [5, 6, 7, 8]:

            color = COLOR_INDEX

        elif start_index in [9, 10, 11, 12] and \
             end_index in [9, 10, 11, 12]:

            color = COLOR_MIDDLE

        elif start_index in [13, 14, 15, 16] and \
             end_index in [13, 14, 15, 16]:

            color = COLOR_RING

        elif start_index in [17, 18, 19, 20] and \
             end_index in [17, 18, 19, 20]:

            color = COLOR_PINKY

        else:

            # Palm connections
            color = COLOR_PALM


        # ---------------------------------------------
        # Draw dark outline first
        # ---------------------------------------------

        cv2.line(
            frame,
            start_point,
            end_point,
            COLOR_POINT_OUTLINE,
            6,
            cv2.LINE_AA
        )


        # ---------------------------------------------
        # Draw colored line
        # ---------------------------------------------

        cv2.line(
            frame,
            start_point,
            end_point,
            color,
            3,
            cv2.LINE_AA
        )


    # ========================================================
    # Draw Landmark Points
    # ========================================================

    for i, point in enumerate(points):

        # ---------------------------------------------
        # Choose point color
        # ---------------------------------------------

        if i == 0:

            color = COLOR_WRIST

        elif i in [1, 2, 3, 4]:

            color = COLOR_THUMB

        elif i in [5, 6, 7, 8]:

            color = COLOR_INDEX

        elif i in [9, 10, 11, 12]:

            color = COLOR_MIDDLE

        elif i in [13, 14, 15, 16]:

            color = COLOR_RING

        else:

            color = COLOR_PINKY


        # ---------------------------------------------
        # Outer circle
        # ---------------------------------------------

        cv2.circle(
            frame,
            point,
            7,
            COLOR_POINT_OUTLINE,
            -1,
            cv2.LINE_AA
        )


        # ---------------------------------------------
        # Inner colored circle
        # ---------------------------------------------

        cv2.circle(
            frame,
            point,
            5,
            color,
            -1,
            cv2.LINE_AA
        )


        # ---------------------------------------------
        # Landmark number
        # ---------------------------------------------

        cv2.putText(
            frame,
            str(i),
            (
                point[0] + 8,
                point[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )


# ============================================================
# Function: Count Extended Fingers
# ============================================================

def count_fingers(hand):

    fingers = 0


    # ========================================================
    # Index
    # ========================================================

    if hand[8].y < hand[6].y:
        fingers += 1


    # ========================================================
    # Middle
    # ========================================================

    if hand[12].y < hand[10].y:
        fingers += 1


    # ========================================================
    # Ring
    # ========================================================

    if hand[16].y < hand[14].y:
        fingers += 1


    # ========================================================
    # Pinky
    # ========================================================

    if hand[20].y < hand[18].y:
        fingers += 1


    # ========================================================
    # Thumb
    #
    # For mirrored webcam view, use X comparison.
    # ========================================================

    if hand[4].x < hand[3].x:
        fingers += 1


    return fingers


# ============================================================
# Hand Landmarker
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # ====================================================
        # Read Camera Frame
        # ====================================================

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read camera frame.")
            break


        # ====================================================
        # Mirror Camera
        # ====================================================

        frame = cv2.flip(frame, 1)


        # ====================================================
        # Frame Dimensions
        # ====================================================

        h, w, _ = frame.shape


        # ====================================================
        # Create Canvas
        # ====================================================

        if canvas is None:

            canvas = np.zeros_like(frame)


        # ====================================================
        # Convert BGR -> RGB
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # Create MediaPipe Image
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ====================================================
        # Timestamp
        # ====================================================

        current_time = int(time.time() * 1000)

        if current_time <= timestamp_ms:

            timestamp_ms += 1

        else:

            timestamp_ms = current_time


        # ====================================================
        # Detect Hand
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # Hand Detected
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # =================================================
            # Convert ALL 21 landmarks to pixel coordinates
            # =================================================

            points = []

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))


            # =================================================
            # Draw Complete Hand Skeleton
            # =================================================

            draw_hand_landmarks(
                frame,
                points
            )


            # =================================================
            # Count Fingers
            # =================================================

            finger_count = count_fingers(hand)


            # =================================================
            # Get Fingertips
            # =================================================

            thumb_point = points[4]
            index_point = points[8]
            middle_point = points[12]
            ring_point = points[16]
            pinky_point = points[20]


            # =================================================
            # Palm Size
            # =================================================

            wrist_point = points[0]
            middle_mcp_point = points[9]

            palm_size = distance(
                wrist_point,
                middle_mcp_point
            )

            palm_size = max(
                palm_size,
                1
            )


            # =================================================
            # Thumb + Index Distance
            # =================================================

            thumb_index_distance = distance(
                thumb_point,
                index_point
            )


            # =================================================
            # Finger Tip Distances
            # =================================================

            index_middle_distance = distance(
                index_point,
                middle_point
            )

            middle_ring_distance = distance(
                middle_point,
                ring_point
            )

            ring_pinky_distance = distance(
                ring_point,
                pinky_point
            )


            # =================================================
            # Paint Gesture
            # =================================================

            paint_threshold = palm_size * 0.45

            paint_mode = (
                thumb_index_distance < paint_threshold
            )


            # =================================================
            # Eraser Gesture
            # =================================================

            eraser_threshold = palm_size * 0.65

            eraser_mode = (

                index_middle_distance < eraser_threshold

                and

                middle_ring_distance < eraser_threshold

                and

                ring_pinky_distance < eraser_threshold
            )


            # =================================================
            # PAINT MODE
            # =================================================

            if paint_mode:

                x, y = index_point


                # ---------------------------------------------
                # Smooth Pointer
                # ---------------------------------------------

                if smooth_point is None:

                    smooth_point = np.array(
                        [x, y],
                        dtype=np.float32
                    )

                else:

                    target = np.array(
                        [x, y],
                        dtype=np.float32
                    )

                    smoothing = 0.35

                    smooth_point = (
                        smooth_point * (1 - smoothing)
                        +
                        target * smoothing
                    )


                smooth_x = int(
                    smooth_point[0]
                )

                smooth_y = int(
                    smooth_point[1]
                )


                # ---------------------------------------------
                # Paint Pointer
                # ---------------------------------------------

                cv2.circle(
                    frame,
                    (smooth_x, smooth_y),
                    10,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA
                )


                # ---------------------------------------------
                # Draw Paint
                # ---------------------------------------------

                if previous_point is not None:

                    cv2.line(
                        canvas,
                        previous_point,
                        (smooth_x, smooth_y),
                        (255, 0, 255),
                        8,
                        cv2.LINE_AA
                    )


                previous_point = (
                    smooth_x,
                    smooth_y
                )


                # ---------------------------------------------
                # Status
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    "PAINT",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    3,
                    cv2.LINE_AA
                )


            # =================================================
            # ERASER MODE
            # =================================================

            elif eraser_mode:

                x, y = index_point


                # ---------------------------------------------
                # Eraser Circle
                # ---------------------------------------------

                cv2.circle(
                    frame,
                    (x, y),
                    30,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )


                # ---------------------------------------------
                # Erase
                # ---------------------------------------------

                cv2.circle(
                    canvas,
                    (x, y),
                    30,
                    (0, 0, 0),
                    -1
                )


                previous_point = None
                smooth_point = None


                # ---------------------------------------------
                # Status
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    "ERASER",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )


            # =================================================
            # No Gesture
            # =================================================

            else:

                previous_point = None
                smooth_point = None


            # =================================================
            # Hand Information Panel
            # =================================================

            # Semi-transparent panel
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (15, 65),
                (230, 125),
                (20, 20, 20),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0,
                frame
            )


            # =================================================
            # Hand Detected Text
            # =================================================

            cv2.putText(
                frame,
                "HAND DETECTED",
                (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                frame,
                f"Fingers: {finger_count}/5",
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # No Hand Detected
        # ====================================================

        else:

            previous_point = None
            smooth_point = None


            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # Combine Canvas + Camera
        # ====================================================

        result_frame = cv2.addWeighted(
            frame,
            1.0,
            canvas,
            1.0,
            0
        )


        # ====================================================
        # Display
        # ====================================================

        cv2.imshow(
            "Virtual Paint - Hand Tracking",
            result_frame
        )


        # ====================================================
        # Keyboard Controls
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # Q = Quit
        if key == ord("q"):

            break


        # C = Clear Canvas
        if key == ord("c"):

            canvas = np.zeros_like(frame)

            previous_point = None
            smooth_point = None


# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()