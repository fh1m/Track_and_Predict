import cv2
import torch
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Clearer font for label
font_path = './assets/iosevka-regular.ttf'
font_size = 22
font = ImageFont.truetype(font_path, font_size)

# Load Pretrained model [TODO: Custom self trained model]
def load_model():
    # Load the YOLOv5 model
    model = torch.hub.load('./yolov7', 'custom', 'yolov7.pt',force_reload=True, source='local',trust_repo=True)
    return model

# what to detect
def detect_objects(model, frame):
    # Perform inference
    results = model(frame)

    # Convert results to pandas DataFrame for easier manipulation
    df = results.pandas().xyxy[0]

    # Filter detections for "bottle" with high confidence
    confidence_threshold = 0.9  # Set a threshold for detection confidence
    bottle_df = df[(df['name'] == 'bottle') & (df['confidence'] > confidence_threshold)]

    # Extract labels and coordinates, adjusting for the DataFrame structure
    labels = bottle_df['name'].to_list()
    cords = bottle_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

    return labels, cords, results.names

# WHat to do after detecting
def draw_boxes(frame, labels, cords, names):
    for label, (xmin, ymin, xmax, ymax) in zip(labels, cords):
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        # Draw rectangle around detected object
        cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
        # Prepare the label text including the object class name
        label_text = f"{label}"
        cv2.putText(frame, label_text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# what tracker to use
def initialize_tracker():
    # Tracker selection
    return cv2.TrackerCSRT_create()

# For better labeling
def draw_text_with_background(image, text, position, font_path, font_size, text_color, bg_color):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    # Manually set background size if textsize is problematic
    bg_width, bg_height = 510, 32  # Adjust based on expected text length and font size
    bg_position = [position[0] - 10 , position[1], position[0] + bg_width, position[1] + bg_height]
    draw.rectangle(bg_position, fill=bg_color)
    draw.text(position, text, fill=text_color, font=font)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image_cv

def calculate_speed(prev_pos, current_pos, prev_time, current_time):
    # Calculate the distance (in pixels) and time elapsed
    dx = current_pos[0] - prev_pos[0]
    dy = current_pos[1] - prev_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    time_elapsed = current_time - prev_time
    speed = distance / time_elapsed if time_elapsed > 0 else 0
    return speed

# main magic
def main():
    model = load_model()
    cap = cv2.VideoCapture(0)  # Use the default webcam
    _, frame = cap.read()
    h, w, _ = frame.shape  # Get the dimensions of the frame
    # changing size
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 3840, 2160)
    # Reference point (0,0) at the center of the frame
    ref_point = (w // 2, h // 2)
    # important vars
    tracker = None
    tracking = False
    detection_interval = 1
    frame_count = 0
    prev_position = None
    prev_time = None
    # start camera feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.circle(frame, ref_point, 5, (0, 0, 255), -1)  # Draw a red circle
        current_time = time.time()
        if not tracking or frame_count % detection_interval == 0:
            # Detect objects only if not currently tracking
            labels, cords, names = detect_objects(model, frame)
            for label, bbox in zip(labels, cords):
                if label == 'bottle':
                    bbox = tuple(map(int, bbox))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.imshow('Frame', frame)
                    cv2.waitKey(1)  # Display the frame for a short moment to confirm the detection
                    tracker = initialize_tracker()
                    tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    tracking = True
                    break  # Track the first detected bottle

        if tracking:
            # Update tracker and get new bounding box
            success, bbox = tracker.update(frame)
            if success:
                # Draw bounding box for tracking
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2)
                current_position = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                if prev_position is not None and prev_time is not None:
                    speed = calculate_speed(prev_position, current_position, prev_time, current_time)
                    # Predict the direction of the bottle
                    direction = "right" if current_position[0] - prev_position[0] > 0 else "left"
                    # Update the previous position and time
                    prev_position = current_position
                    prev_time = current_time
                    # Draw the speed and direction on the frame
                    text = f"Probably move to {direction} with speed of {speed:.2f} px/s"
                    frame = draw_text_with_background(frame, text, (10, h - 30), font_path, font_size, (255, 255, 255), (0, 0, 0))
                else:
                    prev_position = current_position
                    prev_time = current_time
                # Calculate the position of the bottle relative to the reference point
                bottle_center = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                position_relative_to_ref = (bottle_center[0] - ref_point[0], bottle_center[1] - ref_point[1])
                # Determine the direction relative to the reference point
                horizontal_direction = "right" if position_relative_to_ref[0] > 0 else "left"
                vertical_direction = "down" if position_relative_to_ref[1] > 0 else "up"
                # Combine horizontal and vertical directions
                direction = f"{vertical_direction} and {horizontal_direction} of the reference point (0,0) "
                # better label rendered
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                # Label format: "left of the reference point"
                label = f"{direction}"
                # Update the label position to top right corner
                label_position = (10, 0)  # Adjust as needed
                # Use PIL to draw the text
                draw.text(label_position, label, font=font, fill=(28, 21, 28))
                # Convert back to OpenCV image
                frame = draw_text_with_background(
                    image=frame,
                    text=label,
                    position=label_position,
                    font_path=font_path,
                    font_size=22,  # Adjust as needed
                    text_color=(255, 255, 255),  # White text
                    bg_color=(0, 0, 0)  # Black background
            )
                # Optionally, add a label for the object class ("bottle") near the bounding box
                cv2.putText(frame, "Bottle", p1, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
            else:
                # If tracking failed, reset tracking state
                tracking = False
                prev_position = None
                prev_time = None
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        frame_count += 1  # Increment frame counter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

