"""
VisDrone Data Preparation

Converts VisDrone-DET annotations to YOLO format.
This is boilerplate data wrangling - the interesting ML work comes later.

VisDrone annotation format (each line):
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
    
Categories:
    0: ignored regions
    1: pedestrian
    2: people
    3: bicycle
    4: car
    5: van
    6: truck
    7: tricycle
    8: awning-tricycle
    9: bus
    10: motor

YOLO format (each line):
    class_id center_x center_y width height (all normalized 0-1)
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# VisDrone categories we care about (skip 0 = ignored)
VISDRONE_CLASSES = {
    1: 'pedestrian',
    2: 'people', 
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor'
}

# Map VisDrone category IDs to 0-indexed YOLO class IDs
VISDRONE_TO_YOLO = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9   # motor
}


def parse_visdrone_annotation(ann_path: str) -> list[dict]:
    """
    Parse a single VisDrone annotation file.
    
    Returns list of dicts with keys: bbox, category, truncation, occlusion
    """
    annotations = []
    
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            if len(parts) < 8:
                continue
            
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])
            score = int(parts[4])
            category = int(parts[5])
            truncation = int(parts[6])
            occlusion = int(parts[7])
            
            # Skip ignored regions (category 0) and zero-area boxes
            if category == 0 or bbox_width <= 0 or bbox_height <= 0:
                continue
            
            annotations.append({
                'bbox': (bbox_left, bbox_top, bbox_width, bbox_height),
                'category': category,
                'truncation': truncation,
                'occlusion': occlusion
            })
    
    return annotations


def convert_to_yolo_format(
    annotations: list[dict],
    img_width: int,
    img_height: int
) -> list[str]:
    """
    Convert parsed annotations to YOLO format strings.
    
    YOLO format: class_id center_x center_y width height (normalized)
    """
    yolo_lines = []
    
    for ann in annotations:
        category = ann['category']
        
        # Skip if category not in our mapping
        if category not in VISDRONE_TO_YOLO:
            continue
            
        class_id = VISDRONE_TO_YOLO[category]
        bbox_left, bbox_top, bbox_width, bbox_height = ann['bbox']
        
        # Convert to center coordinates
        center_x = bbox_left + bbox_width / 2
        center_y = bbox_top + bbox_height / 2
        
        # Normalize to 0-1
        center_x /= img_width
        center_y /= img_height
        norm_width = bbox_width / img_width
        norm_height = bbox_height / img_height
        
        # Clamp to valid range (some annotations go slightly out of bounds)
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        norm_width = max(0, min(1, norm_width))
        norm_height = max(0, min(1, norm_height))
        
        yolo_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
        )
    
    return yolo_lines


def process_split(
    images_dir: str,
    annotations_dir: str,
    output_images_dir: str,
    output_labels_dir: str
):
    """
    Process a full split (train/val/test), converting annotations to YOLO format.
    
    Creates symlinks for images and writes new label files.
    """
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob('*.jpg'))
    print(f"Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        # Find corresponding annotation file
        ann_path = annotations_dir / f"{img_path.stem}.txt"
        
        if not ann_path.exists():
            print(f"Warning: No annotation for {img_path.name}")
            continue
        
        # Get image dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Parse and convert annotations
        annotations = parse_visdrone_annotation(str(ann_path))
        yolo_lines = convert_to_yolo_format(annotations, img_width, img_height)
        
        # Write YOLO label file
        output_label_path = output_labels_dir / f"{img_path.stem}.txt"
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Symlink image (saves disk space vs copying)
        output_img_path = output_images_dir / img_path.name
        if not output_img_path.exists():
            output_img_path.symlink_to(img_path.resolve())


def create_dataset_yaml(output_dir: str, yaml_path: str):
    """
    Create the dataset.yaml file that Ultralytics expects.
    """
    yaml_content = f"""# VisDrone Dataset (YOLO format)
path: {output_dir}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

# Number of classes
nc: 10
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created {yaml_path}")


def main():
    """
    Main conversion pipeline.
    
    Expected input structure:
        data/raw/
            VisDrone2019-DET-train/
                images/
                annotations/
            VisDrone2019-DET-val/
                images/
                annotations/
            VisDrone2019-DET-test-dev/
                images/
                annotations/
    
    Output structure:
        data/processed/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/
            dataset.yaml
    """
    
    # Adjust these paths to match your setup
    raw_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    splits = {
        'train': 'VisDrone2019-DET-train',
        'val': 'VisDrone2019-DET-val', 
        'test': 'VisDrone2019-DET-test-dev'
    }
    
    for split_name, folder_name in splits.items():
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split...")
        print('='*50)
        
        split_dir = raw_dir / folder_name
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping...")
            continue
        
        process_split(
            images_dir=split_dir / "images",
            annotations_dir=split_dir / "annotations",
            output_images_dir=output_dir / "images" / split_name,
            output_labels_dir=output_dir / "labels" / split_name
        )
    
    # Create dataset.yaml
    create_dataset_yaml(
        output_dir=str(output_dir.resolve()),
        yaml_path=str(output_dir / "dataset.yaml")
    )
    
    print("\nDone! Your dataset is ready at:", output_dir)
    print("Use the dataset.yaml path when training with Ultralytics.")


if __name__ == "__main__":
    main()
