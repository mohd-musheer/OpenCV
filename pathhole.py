from ultralytics import YOLO
import cv2

# Load pothole model
model = YOLO("best.pt")

# Load video
cap = cv2.VideoCapture("videoplayback.mp4")

# Video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Frame delay for original video speed
delay = int(1000 / fps)

print("Video FPS:", fps)
print("Playback delay:", delay, "ms")

# Process YOLO every 2nd frame
frame_skip = 2
frame_count = 0

# Store previous detections
detections = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # -----------------------------------
    # Run YOLO every 2nd frame
    # -----------------------------------

    if frame_count % frame_skip == 0:

        results = model.predict(
            frame,
            conf=0.40,
            imgsz=320,
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                confidence = float(
                    box.conf[0]
                )

                detections.append(
                    (x1, y1, x2, y2, confidence)
                )

    # -----------------------------------
    # Draw latest detections
    # -----------------------------------

    for x1, y1, x2, y2, confidence in detections:

        # Rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        # Text
        cv2.putText(
            frame,
            f"HOLE DETECTED {confidence:.2f}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # -----------------------------------
    # Show video
    # -----------------------------------

    cv2.imshow(
        "Pothole Detection",
        frame
    )

    # -----------------------------------
    # Q = quit
    # -----------------------------------

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()