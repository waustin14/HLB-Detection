import albumentations as A
import cv2
import os

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        scale=(0.5, 1.5),
        translate_px=(-30, 30),
        rotate=(-45, 45),
        p=1.0
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, 
        contrast_limit=0.2, 
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10, 
        sat_shift_limit=15, 
        val_shift_limit=10, 
        p=0.3
    )],
    bbox_params=A.BboxParams(
        format='yolo',
        min_area=2048,
        min_visibility=0.3,
        label_fields=['class_labels']
    )
)

def load_image(image_path):
    """
    Load an image from a file path.
    Args:
        image_path (str): The path to the image file.
    Returns:
        numpy.ndarray: The loaded image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    return image

def load_label(label_path):
    """
    Load labels from a file.
    Args:
        label_path (str): The path to the label file.
    Returns:
        list of tuples: List of bounding boxes in the format (x_center, y_center, width, height).
        list of int: List of class labels corresponding to the bounding boxes.
    """
    bounding_boxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields = line.split(' ')
            if len(fields) != 5:
                raise ValueError(f"Invalid label format in {label_path}: {line}")
            class_labels.append(int(fields[0]))
            bounding_boxes.append(tuple(map(float, fields[1:])))
    return bounding_boxes, class_labels

def augment_image(image, bounding_boxes=None, class_labels=None):
    """"
    "Augment an image and its bounding boxes."
    Args:
        image (numpy.ndarray): The input image.
        bounding_boxes (list of tuples): List of bounding boxes in the format (x_center, y_center, width, height).
        class_labels (list of int): List of class labels corresponding to the bounding boxes.
    Returns:
        numpy.ndarray: The augmented image.
        list of tuples: The augmented bounding boxes in the format (x_center, y_center, width, height).
        list of int: The augmented class labels.
    """
    augmented = augmentation_pipeline(image=image, bboxes=bounding_boxes, class_labels=class_labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_class_labels = [int(cl) for cl in augmented['class_labels']]
    return augmented_image, augmented_bboxes, augmented_class_labels

def save_augmented_image(image, img_path, label_path, bounding_boxes=None, class_labels=None):
    """
    Save the augmented image to the specified path.
    Args:
        image (numpy.ndarray): The augmented image.
        img_path (str): The path where the image will be saved.
        label_path (str): The path where the labels will be saved.
        bounding_boxes (list of tuples): List of bounding boxes in the format (x_center, y_center, width, height).
        class_labels (list of int): List of class labels corresponding to the bounding boxes.
    Returns:
        None
    """
    # Check if the img_path and label_path share the same file name.  Raise ValueError if not.
    if not img_path.endswith(('.jpg', '.JPG', '.png', '.PNG')):
        raise ValueError("Image path must end with '.jpg', '.JPG', '.png', or '.PNG'")
    if not label_path.endswith('.txt'):
        raise ValueError("Label path must end with '.txt'")
    if img_path.split('/')[-1].split('.')[0] != label_path.split('/')[-1].split('.')[0]:
        raise ValueError("Image and label paths must have the same file name")
    # Save the image and labels
    cv2.imwrite(img_path, image)
    if bounding_boxes is not None and class_labels is not None:
        with open(label_path, 'w') as f:
            for bbox, label in zip(bounding_boxes, class_labels):
                f.write(f"{int(label)} {' '.join(map(str, bbox))}\n")

def augment_dataset(image_dir, label_dir, output_dir, num_augmentations=5):
    """
    Augment a dataset of images and labels.
    Args:
        image_dir (str): Directory containing the original images.
        label_dir (str): Directory containing the original labels.
        output_dir (str): Directory where augmented images and labels will be saved.
        num_augmentations (int): Number of augmentations to apply to each image.
    Returns:
        None
    """
    # Create output directories for images and labels
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Iterate through all images in the image directory
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.JPG', '.png', '.PNG')):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.JPG', '.txt').replace('.png', '.txt').replace('.PNG', '.txt'))
            image = load_image(img_path)
            bounding_boxes, class_labels = load_label(label_path)

            # Perform augmentations
            for i in range(num_augmentations):
                try:
                    augmented_image, augmented_bboxes, augmented_class_labels = augment_image(image, bounding_boxes, class_labels)
                    output_img_path = os.path.join(output_dir, 'images', f"{img_file.split('.')[0]}_aug_{i}.jpg")
                    output_label_path = os.path.join(output_dir, 'labels', f"{img_file.split('.')[0]}_aug_{i}.txt")
                    save_augmented_image(augmented_image, output_img_path, output_label_path, augmented_bboxes, augmented_class_labels)
                except Exception as e:
                    print(f"Error during augmentation {i} for {img_file}: {e}")
                    continue