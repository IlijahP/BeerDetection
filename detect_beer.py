import cv2
from ultralytics import YOLO
from collections import deque

# Load the YOLO model
try:
    model = YOLO('best.pt')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'best.pt' is a valid YOLO model file in the current directory.")
    exit()

# Try to open webcam (try different indices if 0 fails)
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# If camera 0 fails, try camera 1
if not cap.isOpened():
    print(f"Camera {camera_index} not available, trying camera 1...")
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open any camera.")
    print("Possible solutions:")
    print("1. Make sure your webcam is connected and not used by another application")
    print("2. Check camera permissions in Windows settings")
    print("3. Try running the script outside of VS Code")
    print("4. Update your webcam drivers")
    exit()

# Test reading a frame to ensure camera works
ret, test_frame = cap.read()
if not ret or test_frame is None:
    print("Error: Camera opened but cannot capture frames.")
    print("This might be a permission issue. Please:")
    print("1. Go to Windows Settings > Privacy & security > Camera")
    print("2. Make sure 'Allow apps to access your camera' is ON")
    print("3. Allow camera access for your Python/Command Prompt application")
    cap.release()
    exit()

# Buffer for detections - keep track of last 5 frames
detection_buffer = deque(maxlen=5)
buffer_threshold = 3  # Need detection in at least 3 of last 5 frames

print(f"Starting live beer detection on camera {camera_index}... Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame. Retrying...")
        continue  # Try again instead of exiting

    # Run YOLO inference on the frame
    results = model(frame)

    # Get current frame detections
    current_detections = set()
    if results and len(results) > 0:
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    current_detections.add(class_name)

    # Add current detections to buffer
    detection_buffer.append(current_detections)

    # Calculate stable detections (present in at least threshold frames)
    if len(detection_buffer) >= buffer_threshold:
        all_detections = set()
        for detections in detection_buffer:
            all_detections.update(detections)

        stable_detections = set()
        for detection in all_detections:
            count = sum(1 for frame_dets in detection_buffer if detection in frame_dets)
            if count >= buffer_threshold:
                stable_detections.add(detection)

        # Filter results to only show stable detections
        filtered_results = []
        if results and len(results) > 0:
            for result in results:
                if result.boxes:
                    # Create new result with only stable detections
                    stable_boxes = []
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls.item())
                        class_name = model.names[class_id]
                        if class_name in stable_detections:
                            stable_boxes.append(box)

                    if stable_boxes:
                        # Create a new result object with filtered boxes
                        new_result = result
                        new_result.boxes = stable_boxes
                        filtered_results.append(new_result)

        # Use filtered results for plotting
        if filtered_results:
            annotated_frame = filtered_results[0].plot()
        else:
            annotated_frame = frame  # Show original frame if no stable detections
    else:
        # Not enough frames in buffer yet, show original frame
        annotated_frame = frame

    # Display the resulting frame
    cv2.imshow('Beer Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

print("Detection stopped.")