import cv2

cap = cv2.VideoCapture("videoplayback.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------
    # 1. Grayscale
    # -----------------------------
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------
    # 2. Blur
    # -----------------------------
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------
    # 3. Threshold
    # -----------------------------
    _, thresh = cv2.threshold(
        gray,
        130,
        255,
        cv2.THRESH_BINARY_INV
    )

    # -----------------------------
    # 4. Find contours
    # -----------------------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # -----------------------------
    # 5. Check every contour
    # -----------------------------
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny objects
        if area < 400:
            continue

        # Ignore extremely large objects
        if area > 20000:
            continue

        # -----------------------------
        # Rectangle around contour
        # -----------------------------
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small rectangles
        if w < 20 or h < 20:
            continue

        # Ignore extremely large rectangles
        if w > 300 or h > 300:
            continue

        # -----------------------------
        # Aspect ratio
        # -----------------------------
        aspect_ratio = w / float(h)

        # Reject very thin lines
        if aspect_ratio > 3.5:
            continue

        if aspect_ratio < 0.25:
            continue

        # -----------------------------
        # Draw rectangle
        # -----------------------------
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        # -----------------------------
        # Text
        # -----------------------------
        cv2.putText(
            frame,
            "HOLE DETECTED",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # -----------------------------
    # Show video
    # -----------------------------
    cv2.imshow(
        "Pothole Detection",
        frame
    )

    # 30 FPS
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()