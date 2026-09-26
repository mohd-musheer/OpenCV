import cv2
import mediapipe as mp
import numpy as np
import time


# ==========================================
# MediaPipe Selfie Segmentation
# ==========================================

mp_selfie = mp.solutions.selfie_segmentation


# ==========================================
# Camera
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")


# ==========================================
# Background settings
# ==========================================

background_mode = "black"

previous_time = time.time()


# ==========================================
# Start segmentation
# ==========================================

with mp_selfie.SelfieSegmentation(
    model_selection=1
) as selfie:

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # ----------------------------------
        # Mirror camera
        # ----------------------------------

        frame = cv2.flip(frame, 1)


        # ----------------------------------
        # Convert BGR → RGB
        # ----------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------
        # Segmentation
        # ----------------------------------

        results = selfie.process(rgb)


        # ----------------------------------
        # Get segmentation mask
        # ----------------------------------

        mask = results.segmentation_mask


        # ----------------------------------
        # Smooth mask
        # ----------------------------------

        mask = cv2.GaussianBlur(
            mask,
            (7, 7),
            0
        )


        # ----------------------------------
        # Threshold
        # ----------------------------------

        mask = np.clip(
            (mask - 0.25) / 0.5,
            0,
            1
        )


        # ----------------------------------
        # Create 3-channel mask
        # ----------------------------------

        mask_3 = np.dstack(
            [mask, mask, mask]
        )


        # ==================================
        # Background
        # ==================================

        if background_mode == "black":

            background = np.zeros_like(frame)


        elif background_mode == "green":

            background = np.zeros_like(frame)

            background[:] = (0, 180, 0)


        elif background_mode == "white":

            background = np.ones_like(frame) * 255


        else:

            background = frame.copy()


        # ==================================
        # Combine foreground + background
        # ==================================

        output = (
            frame * mask_3 +
            background * (1 - mask_3)
        )

        output = output.astype(np.uint8)


        # ==================================
        # FPS
        # ==================================

        current_time = time.time()

        fps = 1 / (current_time - previous_time)

        previous_time = current_time


        cv2.putText(
            output,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            output,
            f"Background: {background_mode}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ==================================
        # Display
        # ==================================

        cv2.imshow(
            "AI Background Removal",
            output
        )


        # ==================================
        # Keyboard controls
        # ==================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("b"):
            background_mode = "black"


        elif key == ord("g"):
            background_mode = "green"


        elif key == ord("w"):
            background_mode = "white"


        elif key == ord("n"):
            background_mode = "none"


        elif key == ord("q"):
            break


# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()