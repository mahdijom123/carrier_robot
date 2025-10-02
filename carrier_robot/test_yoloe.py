import cv2
import os
import json
from ultralytics import YOLOWorld

# -----------------------
# Load YOLO-World
# -----------------------
yolo_model = YOLOWorld("yolov8s-worldv2.pt")

# -----------------------
# JSON reload helpers
# -----------------------
json_path = os.path.join(os.path.dirname(__file__), "prompts.json")
last_json_mtime = 0
labels = []
short_labels = []
prompt_map = {}

def load_prompts():
    global last_json_mtime, labels, short_labels, prompt_map
    try:
        mtime = os.path.getmtime(json_path)
        if mtime != last_json_mtime:
            with open(json_path, "r", encoding="utf-8") as f:
                prompt_map = json.load(f)
            labels = list(prompt_map.keys())
            short_labels = list(prompt_map.values())
            yolo_model.set_classes(labels)
            last_json_mtime = mtime
            print("[INFO] Prompts reloaded:", labels)
    except Exception as e:
        print("[ERROR] Failed to load prompts:", e)

# -----------------------
# Initialize Webcam
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Starting YOLO tracking with vector + distance... Press 'q' to quit.")

CONF_THRESHOLD = 0.4
last_results = None
persist_frames = 10
miss_count = 0

# -----------------------
# Choose target
# -----------------------
TARGET = "water bottle"   # <-- set your target item here
CM_PER_PIXEL = 0.1        # <-- calibration factor: adjust this after testing

# -----------------------
# Main Loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    load_prompts()

    # Run YOLO detection on every frame
    results = yolo_model.predict(frame, verbose=False, conf=CONF_THRESHOLD, imgsz=480)
    if results and len(results[0].boxes) > 0:
        found_target = False
        for box in results[0].boxes:
            cls_idx = int(box.cls)
            label = labels[cls_idx]
            if label == TARGET:
                last_results = [box]
                found_target = True
                miss_count = 0
                break
        if not found_target:
            miss_count += 1
    else:
        miss_count += 1

    annotated_frame = frame.copy()

    if last_results and miss_count < persist_frames:
        for box in last_results:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
            conf = float(box.conf)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{TARGET} ({conf:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # Compute centers
            frame_center = (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2)
            obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Direction vector (center - object)
            dx = frame_center[0] - obj_center[0]
            dy = frame_center[1] - obj_center[1]

            # Length of vector (pixels → cm)
            vector_length_px = (dx**2 + dy**2) ** 0.5
            vector_length_cm = vector_length_px * CM_PER_PIXEL

            # Horizontal correction (left-right movement in cm)
            move_cm = dx * CM_PER_PIXEL
            if abs(move_cm) > 1:  # threshold of 1 cm
                direction = "RIGHT" if move_cm < 0 else "LEFT"
                instruction = f"Move {direction} by {abs(move_cm):.1f} cm"
            else:
                instruction = "Centered ✔"

            # Draw arrow (object → center)
            cv2.circle(annotated_frame, frame_center, 5, (0, 255, 255), -1)  # Yellow = center
            cv2.circle(annotated_frame, obj_center, 5, (0, 0, 255), -1)      # Red = object
            cv2.arrowedLine(annotated_frame, obj_center, frame_center, (255, 0, 0), 2)

            # Display instruction
            cv2.putText(annotated_frame, instruction, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Debug print
            print(f"[VECTOR] dx={dx}, dy={dy}, len={vector_length_px:.1f}px ≈ {vector_length_cm:.1f}cm | {instruction}")

    cv2.imshow("YOLO Tracking + Distance", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
