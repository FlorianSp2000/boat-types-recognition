""" Helper and util functions"""
import cv2
import os

def plot_gt_bounding_box(img_path: str):
    """
    Plot ground truth bounding box on an image.

    :param img_path: Path to the image file.
    """
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image {img_path}")
        return

    # Get the image dimensions
    img_height, img_width = img.shape[:2]

    # Derive the corresponding label file path
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

    if not os.path.exists(label_path):
        print(f"Could not find label file {label_path}")
        return

    # Read the bounding box coordinates from the label file
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert normalized coordinates to pixel coordinates
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw the bounding box
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add a label for the bounding box
            cv2.putText(img, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the image
    cv2.imshow("Image with Ground Truth Bounding Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
plot_gt_bounding_box("train_dataset/images/buoy_alaska-ocean-warning-light-sea-2574393_jpg.rf.67cf23f5f470d5a05b08c87d7391e126.jpg")
