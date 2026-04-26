import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()
fps = 0.0
alpha = 0.9  # smoothing factor for EMA

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        instant_fps = 1.0 / dt
        fps = alpha * fps + (1 - alpha) * instant_fps if fps > 0 else instant_fps

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Camera FPS Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final measured FPS: {fps:.1f}")
