import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_contrast_enhancement(original_img, enhanced_img, img_name, output_dir):
    """Generate a professional contrast enhancement comparison figure"""
    plt.figure(figsize=(12, 8))
    
    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image', fontsize=10)
    plt.axis('off')
    
    # Enhanced Image
    plt.subplot(2, 2, 2)
    plt.imshow(enhanced_img, cmap='gray')
    plt.title('Enhanced Image', fontsize=10)
    plt.axis('off')
    
    # Original Histogram
    plt.subplot(2, 2, 3)
    plt.hist(original_img.flatten(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('Original Histogram', fontsize=10)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Enhanced Histogram
    plt.subplot(2, 2, 4)
    plt.hist(enhanced_img.flatten(), 256, [0, 256], color='green', alpha=0.7)
    plt.title('Enhanced Histogram', fontsize=10)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"contrast_enhancement_{Path(img_name).stem}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_images(input_dir, output_dir, method='clahe'):
    """
    Process images and generate contrast enhancement figures
    Available methods: 'clahe' or 'histeq'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No valid images found in {input_dir}")
        return
    
    for img_name in image_files:
        img_path = Path(input_dir) / img_name
        original_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if original_img is not None:
            # Apply enhancement
            if method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_img = clahe.apply(original_img)
            else:  # Regular histogram equalization
                enhanced_img = cv2.equalizeHist(original_img)
            
            # Save enhanced image
            cv2.imwrite(str(output_dir / img_name), enhanced_img)
            
            # Generate and save comparison figure
            plot_contrast_enhancement(original_img, enhanced_img, img_name, output_dir)
            
            print(f"Processed: {img_name}")
        else:
            print(f"Could not read image: {img_name}")

if __name__ == "__main__":
    # Configuration - Update these paths to match your project structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE_DIR, "data", "train", "tumor")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", "contrast_enhancement")
    METHOD = 'clahe'  # Options: 'clahe' or 'histeq'
    
    # Verify paths exist
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found at {INPUT_DIR}")
        print("Please ensure:")
        print(f"1. The directory 'data/train/tumor' exists in your project")
        print(f"2. The directory contains image files (.png, .jpg, .jpeg)")
        print("\nCurrent project structure should look like:")
        print("tumor-classifier-main/")
        print("├── data/")
        print("│   └── train/")
        print("│       └── tumor/  <-- Your images should be here")
        print("├── src/")
        print("│   └── preprocess.py")
        print("└── ...")
        exit(1)
    
    # Process images
    print(f"Starting image processing from: {INPUT_DIR}")
    process_images(INPUT_DIR, OUTPUT_DIR, method=METHOD)
    print(f"\nProcessing complete! Results saved to: {OUTPUT_DIR}")