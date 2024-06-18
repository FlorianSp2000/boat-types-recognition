""" detect script based on ultralytics yolov10 model"""
import os
import sys
import argparse
import cv2
from ultralytics import YOLO

def detect_boats_in_image(image_path: str, model_path: str) -> None:
    """
    Detect boats in an image using the YOLO model.

    :param image_path: Path to the image file.
    :param model_path: Path to the trained YOLO model file.
    """
    model = YOLO(model_path)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return

    results = model(image)

    # Draw bounding boxes and labels on the image
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result[:6]
        label = model.names[int(class_id)]
        if label == 'boat':  # Assuming 'boat' is the label for boats
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or display the image
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_detected{ext}"
    cv2.imwrite(output_path, image)
    print(f"Detection results saved to {output_path}")
    cv2.imshow('Detected Boats', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_boats_in_video(video_path: str, model_path: str) -> None:
    """
    Detect boats in a video using the YOLO model.

    :param video_path: Path to the video file.
    :param model_path: Path to the trained YOLO model file.
    """
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = 'output_video.avi'

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results.xyxy[0]:
            x1, y1, x2, y2, confidence, class_id = result[:6]
            label = model.names[int(class_id)]
            if label == 'boat':  # Assuming 'boat' is the label for boats
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

        # Display the frame
        cv2.imshow('Boat Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detect boats in images or videos using a trained OD model.")
    parser.add_argument('model', type=str, help="Path to the trained YOLO model (.pt) file")
    parser.add_argument('media', type=str, help="Path to the image or video file")
    parser.add_argument('--type', type=str, choices=['image', 'video'], required=True, help="Type of the media (image or video)")
    # TODO: Add argument to specify the output directory, whether to save the results to a file or display them
    args = parser.parse_args()

    if args.type == 'image':
        detect_boats_in_image(args.media, args.model)
    elif args.type == 'video':
        detect_boats_in_video(args.media, args.model)

if __name__ == "__main__":
    main()

# python detect.py path/to/your/model.pt path/to/your/image.jpg --type image
# python detect.py path/to/your/model.pt path/to/your/video.mp4 --type video 
