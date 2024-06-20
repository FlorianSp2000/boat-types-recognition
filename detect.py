""" detect script based on ultralytics yolov10 model"""
import os
import argparse
import cv2

from ultralytics import YOLO

from constants import CLASS_COLOR_MAP

def detect_boats_in_image(image_path: str, model_path: str, save_image: bool) -> None:
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
    
    result = model(image)[0]

    # Draw bounding boxes and labels on the image
    for i in range(len(result.boxes.xyxy)):
        x1, y1, x2, y2 = result.boxes.xyxy[i]
        confidence, class_id = result.boxes.conf[i], result.boxes.cls[i]
        label = model.names[int(class_id)]

        bounding_box_color = CLASS_COLOR_MAP.get(label, (0, 255, 0))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), bounding_box_color, 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bounding_box_color, 2)

    # Save or display the image
    if save_image:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detected{ext}"
        cv2.imwrite(output_path, image)
        print(f"Detection results saved to {output_path}")
    cv2.imshow('Detected Boats', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_boats_in_video(video_path: str, model_path: str, save_video: bool) -> None:
    """
    Detect boats in a video using ultralytics YOLO model.

    :param video_path: Path to the video file.
    :param model_path: Path to the trained YOLO model file.
    :param save_video: Whether to save the processed video or not.
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_video:
        # Prepare output video
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        # Save video as same file type as input video
        base_name, ext = os.path.splitext(os.path.basename(video_path))
        
        output_file = f"{base_name}_detected_{model_name}.{ext}"
        if not os.path.exists('evaluation_results'):
            os.makedirs('evaluation_results')
        output_path = os.path.join('evaluation_results', output_file)

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        # Draw bounding boxes and labels on the frame
        for i in range(len(results.boxes.xyxy)):
            x1, y1, x2, y2 = results.boxes.xyxy[i]
            confidence, class_id = results.boxes.conf[i], results.boxes.cls[i]
            label = model.names[int(class_id)]

            bounding_box_color = CLASS_COLOR_MAP.get(label, (0, 255, 0))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bounding_box_color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bounding_box_color, 2)

        if save_video:
            out.write(frame)

        # Display the frame
        cv2.imshow('Boat Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    if save_video:
        out.release()
        print(f"Processed video saved to {output_path}")
    cv2.destroyAllWindows()
    

def main():
    parser = argparse.ArgumentParser(description="Detect boats in images or videos using a trained OD model.")
    parser.add_argument('model', type=str, help="Path to the trained YOLO model (.pt) file")
    parser.add_argument('media', type=str, help="Path to the image or video file")
    parser.add_argument('--type', type=str, choices=['image', 'video'], required=True, help="Type of the media (image or video)")
    parser.add_argument('--save', action='store_true', help="Flag indicating whether to save the results to a file")

    # TODO: Add argument to specify the output directory
    args = parser.parse_args()

    if args.type == 'image':
        detect_boats_in_image(args.media, args.model, args.save)
    elif args.type == 'video':
        detect_boats_in_video(args.media, args.model, args.save)

if __name__ == "__main__":
    main()
