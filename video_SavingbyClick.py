import cv2

# -----------------------------
# Open Camera
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get camera resolution
frame_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Resolution: {frame_w} x {frame_h}")

# -----------------------------
# Recording settings
# -----------------------------
codec = cv2.VideoWriter_fourcc(*"XVID")
fps = 30

recorder = None
recording = False

# -----------------------------
# Main camera loop
# -----------------------------
while True:

    success, frame = camera.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # -------------------------
    # Start recording
    # -------------------------
    if recording:

        # Write current frame to video
        recorder.write(frame)

        # Display recording status
        cv2.putText(
            frame,
            "RECORDING...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    else:

        # Display instruction
        cv2.putText(
            frame,
            "Press S to Start Recording",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show camera
    cv2.imshow("Live Camera", frame)

    # Wait for keyboard input
    key = cv2.waitKey(1) & 0xFF

    # -------------------------
    # Start recording
    # -------------------------
    if key == ord("s") and not recording:

        recorder = cv2.VideoWriter(
            "MyVideo.avi",
            codec,
            fps,
            (frame_w, frame_h)
        )

        recording = True

        print("Recording started...")

    # -------------------------
    # Stop recording
    # -------------------------
    elif key == ord("r") and recording:

        recording = False

        recorder.release()
        recorder = None

        print("Recording stopped.")
        print("Video saved as MyVideo.avi")

    # -------------------------
    # Quit
    # -------------------------
    elif key == ord("q"):

        print("Exiting...")
        break


# -----------------------------
# Cleanup
# -----------------------------

camera.release()

if recorder is not None:
    recorder.release()

cv2.destroyAllWindows()