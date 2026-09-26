import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# MEDIAPIPE POSE LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "pose_landmarker.task"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Maximum number of people to detect
MAX_PEOPLE = 4


# ============================================================
# COLORS - BGR
#
# Each detected person gets a different color.
# ============================================================

PERSON_COLORS = [

    (255, 255, 0),      # Cyan

    (255, 0, 255),      # Magenta

    (0, 255, 255),      # Yellow

    (80, 255, 100),     # Green

]


# ============================================================
# POSE CONNECTIONS
#
# MediaPipe Pose = 33 landmarks
# ============================================================

POSE_CONNECTIONS = [

    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Left hand
    (15, 17),
    (15, 19),
    (15, 21),

    # Right hand
    (16, 18),
    (16, 20),
    (16, 22),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28),

    # Left foot
    (27, 29),
    (29, 31),

    # Right foot
    (28, 30),
    (30, 32),
]


# ============================================================
# MEDIAPIPE OPTIONS
# ============================================================

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    # IMPORTANT:
    # Detect multiple people
    num_poses=MAX_PEOPLE,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0

previous_time = time.time()

fps = 0


# ============================================================
# DRAW PERSON
# ============================================================

def draw_person(
    image,
    pose,
    person_number,
    color
):

    h, w = image.shape[:2]


    # ========================================================
    # Convert landmarks to pixel coordinates
    # ========================================================

    points = []


    for landmark in pose:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        points.append(
            (x, y)
        )


    # ========================================================
    # Draw skeleton
    # ========================================================

    for start, end in POSE_CONNECTIONS:

        if start >= len(points):
            continue

        if end >= len(points):
            continue


        p1 = points[start]

        p2 = points[end]


        # ----------------------------------------------------
        # Thin dark outline
        # ----------------------------------------------------

        cv2.line(

            image,

            p1,

            p2,

            (20, 20, 20),

            4,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Main thin colored line
        # ----------------------------------------------------

        cv2.line(

            image,

            p1,

            p2,

            color,

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # Draw landmarks
    # ========================================================

    for point in points:

        # Small dark outline
        cv2.circle(

            image,

            point,

            5,

            (20, 20, 20),

            -1,

            cv2.LINE_AA
        )


        # Small colored point
        cv2.circle(

            image,

            point,

            3,

            color,

            -1,

            cv2.LINE_AA
        )


    # ========================================================
    # Person label
    # ========================================================

    # Use nose as label position
    nose = points[0]

    label_x = nose[0] + 12

    label_y = nose[1] - 15


    cv2.putText(

        image,

        f"PERSON {person_number}",

        (
            label_x,
            label_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        color,

        1,

        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        # ====================================================
        # READ CAMERA
        #
        # Camera is used only for detection.
        # It is NEVER displayed.
        # ====================================================

        ret, camera_frame = cap.read()


        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break


        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        camera_frame = cv2.flip(
            camera_frame,
            1
        )


        # ====================================================
        # FRAME SIZE
        # ====================================================

        h, w = camera_frame.shape[:2]


        # ====================================================
        # BLACK DISPLAY
        # ====================================================

        display = np.zeros_like(
            camera_frame
        )


        # Slightly dark background
        display[:] = (
            3,
            2,
            8
        )


        # ====================================================
        # CAMERA -> RGB
        # ====================================================

        rgb = cv2.cvtColor(

            camera_frame,

            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        current_time = int(
            time.time() * 1000
        )


        if current_time <= timestamp:

            timestamp += 1

        else:

            timestamp = current_time


        # ====================================================
        # DETECT MULTIPLE PEOPLE
        # ====================================================

        result = landmarker.detect_for_video(

            mp_image,

            timestamp
        )


        # ====================================================
        # DRAW ALL DETECTED PEOPLE
        # ====================================================

        people_count = 0


        if result.pose_landmarks:

            people_count = len(
                result.pose_landmarks
            )


            for index, pose in enumerate(
                result.pose_landmarks
            ):

                # Select person's color
                color = PERSON_COLORS[
                    index % len(PERSON_COLORS)
                ]


                draw_person(

                    display,

                    pose,

                    index + 1,

                    color
                )


        # ====================================================
        # TOP STATUS
        # ====================================================

        cv2.putText(

            display,

            "MULTI-PERSON POSE",

            (25, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 0),

            2,

            cv2.LINE_AA
        )


        # ====================================================
        # PEOPLE COUNT
        # ====================================================

        cv2.putText(

            display,

            f"PEOPLE: {people_count}",

            (25, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        delta = (
            current_time
            -
            previous_time
        )


        if delta > 0:

            instant_fps = 1 / delta

            fps = (

                fps * 0.9

                +

                instant_fps * 0.1
            )


        previous_time = current_time


        cv2.putText(

            display,

            f"FPS: {int(fps)}",

            (25, 100),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (180, 180, 180),

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # SIMPLE BORDER
        # ====================================================

        cv2.rectangle(

            display,

            (8, 8),

            (w - 8, h - 8),

            (80, 40, 100),

            1
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(

            "Multi-Person Pose Studio",

            display
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # Q = Quit
        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()