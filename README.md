# vehical-tracking-
YOLOv8 Model for Vehicle Detection:

YOLOv8 is used to detect different types of vehicles (car, motorbike, truck, and bus) in a video.

For each detected vehicle, it creates bounding boxes and tracks the confidence level of each detection.

Deep SORT for Tracking:

After detecting vehicles, Deep SORT is used to assign unique IDs to each vehicle and track their movement across video frames.

The DeepSort.update() method matches current bounding boxes to previously tracked objects, assigning the same ID to objects that have not moved significantly, simulating object persistence over multiple frames.

Counting Vehicles:

A predefined line is drawn on the video frame, and the vehicles' movement across this line is tracked.

The script counts how many vehicles move "in" and "out" based on their crossing of this line, updating the counts for each vehicle type (car, motorbike, truck, bus).

Output:

The processed video is saved, where each vehicle's bounding box and ID are displayed along with the vehicle counts (in and out).

The script also displays the counts of each vehicle type in real-time on the video frames.

deep_sort.py:
Deep SORT Class:

This script defines the DeepSort class, which is used to manage and track vehicles through unique IDs.

It takes a list of bounding boxes (bboxes), and for each bounding box, it tries to match it to existing tracked objects based on spatial proximity.

If no match is found, a new ID is assigned to the vehicle.

It then updates the tracking list with the new bounding boxes and IDs, which are used in the main image_tracking.py script.

Integration:
Vehicle Detection: YOLOv8 identifies and detects vehicles.

Tracking: Deep SORT tracks these vehicles across frames, assigning unique IDs.

Counting: The script counts vehicles based on their crossing of a predefined line.

Video Output: The processed frames, showing detected vehicles and their counts, are saved in an output video file.
