"""
Data Exploration Utilities

Helpers for understanding the VisDrone dataset before training.
Use these in your exploration notebook.
"""

import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


# Class names for display
CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]


def load_yolo_labels(labels_dir: str) -> list[dict]:
    """
    Load all YOLO-format labels from a directory.
    
    Returns list of dicts with 'file', 'class_id', 'cx', 'cy', 'w', 'h'
    """
    labels_dir = Path(labels_dir)
    all_labels = []
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                all_labels.append({
                    'file': label_file.stem,
                    'class_id': int(parts[0]),
                    'cx': float(parts[1]),
                    'cy': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                })
    
    return all_labels


def get_class_distribution(labels: list[dict]) -> dict:
    """Count objects per class."""
    counts = Counter(label['class_id'] for label in labels)
    return {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}


def plot_class_distribution(labels: list[dict], title: str = "Class Distribution"):
    """Bar chart of class frequencies."""
    dist = get_class_distribution(labels)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(dist.keys(), dist.values())
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, dist.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def compute_object_sizes(
    labels: list[dict],
    images_dir: str,
    sample_size: int = 1000
) -> list[dict]:
    """
    Compute actual pixel sizes of objects.
    
    Need image dimensions to convert normalized coords to pixels.
    Samples images if dataset is large.
    """
    images_dir = Path(images_dir)
    
    # Group labels by file
    labels_by_file = {}
    for label in labels:
        fname = label['file']
        if fname not in labels_by_file:
            labels_by_file[fname] = []
        labels_by_file[fname].append(label)
    
    # Sample files if too many
    files = list(labels_by_file.keys())
    if len(files) > sample_size:
        files = np.random.choice(files, sample_size, replace=False)
    
    sizes = []
    for fname in files:
        # Find image file
        img_path = images_dir / f"{fname}.jpg"
        if not img_path.exists():
            continue
            
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        for label in labels_by_file[fname]:
            pixel_w = label['w'] * img_w
            pixel_h = label['h'] * img_h
            area = pixel_w * pixel_h
            
            sizes.append({
                'class_id': label['class_id'],
                'pixel_w': pixel_w,
                'pixel_h': pixel_h,
                'area': area,
                'max_dim': max(pixel_w, pixel_h)
            })
    
    return sizes


def categorize_by_coco_size(sizes: list[dict]) -> dict:
    """
    Categorize objects by COCO size definitions.
    
    COCO uses area thresholds:
        small: area < 32^2 = 1024
        medium: 32^2 <= area < 96^2 = 9216  
        large: area >= 96^2
    """
    categories = {'small': 0, 'medium': 0, 'large': 0}
    
    for s in sizes:
        area = s['area']
        if area < 1024:
            categories['small'] += 1
        elif area < 9216:
            categories['medium'] += 1
        else:
            categories['large'] += 1
    
    return categories


def plot_size_distribution(sizes: list[dict], title: str = "Object Size Distribution"):
    """Histogram of object sizes with COCO thresholds marked."""
    areas = [s['area'] for s in sizes]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    ax1 = axes[0]
    ax1.hist(areas, bins=100, edgecolor='black', alpha=0.7)
    ax1.axvline(x=1024, color='r', linestyle='--', label='small/medium (32²)')
    ax1.axvline(x=9216, color='g', linestyle='--', label='medium/large (96²)')
    ax1.set_xlabel('Area (pixels²)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title} (linear scale)')
    ax1.legend()
    
    # Log scale (better for seeing small objects)
    ax2 = axes[1]
    ax2.hist(areas, bins=np.logspace(0, 6, 100), edgecolor='black', alpha=0.7)
    ax2.axvline(x=1024, color='r', linestyle='--', label='small/medium (32²)')
    ax2.axvline(x=9216, color='g', linestyle='--', label='medium/large (96²)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Area (pixels²) - log scale')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{title} (log scale)')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def visualize_detections(
    img_path: str,
    label_path: str,
    figsize: tuple = (12, 8)
):
    """
    Visualize an image with its ground truth boxes.
    
    Useful for sanity checking your data conversion.
    """
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            
            # Convert to pixel coordinates
            px_cx = cx * img_w
            px_cy = cy * img_h
            px_w = w * img_w
            px_h = h * img_h
            
            # Convert center to top-left for matplotlib
            px_left = px_cx - px_w / 2
            px_top = px_cy - px_h / 2
            
            rect = patches.Rectangle(
                (px_left, px_top), px_w, px_h,
                linewidth=2, edgecolor=colors[class_id],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(
                px_left, px_top - 2,
                CLASS_NAMES[class_id],
                fontsize=8, color='white',
                bbox=dict(boxstyle='round', facecolor=colors[class_id], alpha=0.8)
            )
    
    ax.set_title(Path(img_path).name)
    ax.axis('off')
    plt.tight_layout()
    return fig


def sample_and_visualize(
    images_dir: str,
    labels_dir: str,
    n_samples: int = 6,
    seed: int = 42
):
    """Visualize random samples from the dataset."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    np.random.seed(seed)
    
    label_files = list(labels_dir.glob('*.txt'))
    samples = np.random.choice(label_files, min(n_samples, len(label_files)), replace=False)
    
    for label_path in samples:
        img_path = images_dir / f"{label_path.stem}.jpg"
        if img_path.exists():
            visualize_detections(str(img_path), str(label_path))
            plt.show()


# Quick stats function for notebooks
def dataset_summary(labels_dir: str, images_dir: str = None):
    """Print a quick summary of the dataset."""
    labels = load_yolo_labels(labels_dir)
    
    print(f"Total objects: {len(labels):,}")
    print(f"Total images with labels: {len(set(l['file'] for l in labels)):,}")
    print()
    
    print("Class distribution:")
    dist = get_class_distribution(labels)
    for cls, count in dist.items():
        pct = 100 * count / len(labels)
        print(f"  {cls}: {count:,} ({pct:.1f}%)")
    print()
    
    if images_dir:
        print("Computing object sizes (sampling 1000 images)...")
        sizes = compute_object_sizes(labels, images_dir, sample_size=1000)
        size_cats = categorize_by_coco_size(sizes)
        
        total = sum(size_cats.values())
        print("Size distribution (COCO definitions):")
        for cat, count in size_cats.items():
            pct = 100 * count / total
            print(f"  {cat}: {count:,} ({pct:.1f}%)")
