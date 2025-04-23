import cv2
import time
import os
import importlib.util
from ultralytics import YOLO  # Import YOLOv8 from ultralytics

# Correct path to deep_sort.py within the subfolder
deep_sort_path = os.path.join(os.path.dirname(__file__), 'deep_sort', 'deep_sort.py')

# Load the deep_sort module dynamically from the specified path
if os.path.exists(deep_sort_path):
    spec = importlib.util.spec_from_file_location("deep_sort", deep_sort_path)
    deep_sort_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deep_sort_module)
    DeepSort = deep_sort_module.DeepSort
else:
    raise FileNotFoundError(f"deep_sort.py not found in {deep_sort_path}. Please ensure it exists.")

# Load YOLOv8 model
yolo = YOLO("yolov8n.pt")  # Using the YOLOv8 nano model for speed

# Initialize Deep SORT for tracking
deepsort = DeepSort(model_path='ckpt.t7')  # Placeholder model path

# Line position for counting
line_position_y = 550  # Y-coordinate for the counting line

# Initialize counters for each type of vehicle
vehicle_counts = {
    "car": {"in": 0, "out": 0},
    "motorbike": {"in": 0, "out": 0},
    "truck": {"in": 0, "out": 0},
    "bus": {"in": 0, "out": 0}
}

# YOLOv8 class indices for different types of vehicles
class_to_vehicle_type = {
    2: "car",       # Car
    3: "motorbike", # Motorbike
    5: "bus",       # Bus
    7: "truck"      # Truck
}

# Track vehicles that have already been counted
counted_vehicles = {}

# Video capture
cap = cv2.VideoCapture('video.mp4')  # Replace with your video file path

# Get video properties for saving output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object for saving output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
output_video = cv2.VideoWriter('output_detected_video.mp4', fourcc, frame_rate, (frame_width, frame_height))

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 detection
    results = yolo(frame)  # Forward the frame to the YOLOv8 model
    bboxes = []
    detected_classes = []  # List to hold the classes of detected objects

    for result in results[0].boxes.data:  # Iterate through detections
        x1, y1, x2, y2, conf, cls = result
        cls = int(cls.item())

        # Check if the detected class is a vehicle type we want to count
        if cls in class_to_vehicle_type:
            vehicle_type = class_to_vehicle_type[cls]
            bboxes.append((x1.item(), y1.item(), x2.item(), y2.item(), conf.item()))
            detected_classes.append((vehicle_type, x1.item(), y1.item(), x2.item(), y2.item(), conf.item()))

            # Draw YOLO detection for debugging
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{vehicle_type.capitalize()} Conf: {conf:.2f}', 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Perform tracking with Deep SORT if there are detections
    if bboxes:
        tracked_outputs = deepsort.update(bboxes, frame)

        # Draw the counting line
        cv2.line(frame, (0, line_position_y), (frame_width, line_position_y), (0, 255, 0), 2)

        # Process tracking results for each vehicle
        for idx, (x1, y1, x2, y2, track_id, conf) in enumerate(tracked_outputs):
            # Determine the type of vehicle using the detected class
            vehicle_type = detected_classes[idx][0] if idx < len(detected_classes) else "unknown"

            # Calculate the center y-coordinate of the bounding box
            center_y = (y1 + y2) / 2

            # Check if the vehicle crosses the line in either direction
            if track_id in counted_vehicles:
                prev_center_y = counted_vehicles[track_id]["last_y"]
                
                # Count when vehicle moves from above to below the line (incoming)
                if prev_center_y < line_position_y and center_y > line_position_y:
                    vehicle_counts[vehicle_type]["in"] += 1
                    print(f"{vehicle_type.capitalize()} ID {track_id} counted coming in. Total {vehicle_type.capitalize()} incoming count: {vehicle_counts[vehicle_type]['in']}")

                # Count when vehicle moves from below to above the line (outgoing)
                elif prev_center_y > line_position_y and center_y < line_position_y:
                    vehicle_counts[vehicle_type]["out"] += 1
                    print(f"{vehicle_type.capitalize()} ID {track_id} counted going out. Total {vehicle_type.capitalize()} outgoing count: {vehicle_counts[vehicle_type]['out']}")

                # Update last known y position
                counted_vehicles[track_id]["last_y"] = center_y
            else:
                # If the vehicle is new, initialize its tracking info
                counted_vehicles[track_id] = {"last_y": center_y}

            # Draw bounding box around detected vehicles
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'{vehicle_type.capitalize()} ID {track_id}', 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display counts for each type of vehicle on the frame
    y_offset = 50
    for vehicle_type, counts in vehicle_counts.items():
        cv2.putText(frame, f'{vehicle_type.capitalize()} Incoming: {counts["in"]}  Outgoing: {counts["out"]}', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 30  # Move each count display down by 30 pixels

    # Write the frame to the output video
    output_video.write(frame)

    # Show frame
    cv2.imshow('Vehicle Tracking and Counting by Type', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
output_video.release()
cv2.destroyAllWindows()

print("Processed video saved as 'output_detected_video.mp4'")