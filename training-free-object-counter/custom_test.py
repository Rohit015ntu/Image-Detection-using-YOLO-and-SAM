#!/usr/bin/env python3
"""
Custom image testing script for training-free object counter
Test on hibiscus flowers or any custom image
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

# Import the necessary modules from the repository
from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from shi_segment_anything import sam_model_registry
import clip
from utils import *

def load_models(device='cuda:0'):
    """Load SAM and CLIP models"""
    print("Loading models...")
    
    # Load SAM model
    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    return mask_generator, clip_model, preprocess

def count_objects_in_image(image_path, object_name="flower", device='cuda:0'):
    """
    Count objects in a custom image
    
    Args:
        image_path: Path to your image
        object_name: What to count (e.g., "flower", "hibiscus flower", "red flower")
        device: CUDA device
    """
    
    # Load models
    mask_generator, clip_model, preprocess = load_models(device)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks using SAM with reference prompt
    print("Generating masks with SAM...")
    # Use the center of the image as a reference point prompt
    h, w = image_rgb.shape[:2]
    ref_prompt = [[w//2, h//2]]  # Center point as reference
    
    try:
        masks = mask_generator.generate(image_rgb, ref_prompt)
    except Exception as e:
        print(f"Error with ref_prompt, trying alternative approach: {e}")
        # Try with box prompt instead
        ref_prompt = [[w//4, h//4, 3*w//4, 3*h//4]]  # Box covering most of the image
        masks = mask_generator.generate(image_rgb, ref_prompt)
    
    print(f"Generated {len(masks)} masks")
    
    # Create text prompt for CLIP
    text_prompt = f"a photo of a {object_name}"
    text_tokens = clip.tokenize([text_prompt]).to(device)
    
    # Get text features
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Score each mask
    valid_masks = []
    scores = []
    
    print("Scoring masks with CLIP...")
    for mask_data in masks:
        mask = mask_data['segmentation']
        
        # Extract masked region
        masked_image = image_rgb.copy()
        masked_image[~mask] = 0  # Set background to black
        
        # Convert to PIL and preprocess for CLIP
        pil_image = Image.fromarray(masked_image)
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity_tensor = (image_features @ text_features.T)
            similarity = float(similarity_tensor.squeeze().cpu().numpy())
            scores.append(similarity)
    
    # Filter masks based on similarity threshold
    threshold = 0.25  # Adjust this threshold as needed
    for i, score in enumerate(scores):
        if score > threshold:
            valid_masks.append(masks[i])
    
    count = len(valid_masks)
    print(f"Found {count} {object_name}(s)")
    
    # Visualize results
    visualize_results(image_rgb, valid_masks, count, object_name)
    
    return count, valid_masks

def visualize_results(image, masks, count, object_name):
    """Visualize the counting results"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Masks overlay
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    for mask_data in masks:
        mask = mask_data['segmentation']
        color = np.random.rand(3)
        plt.imshow(mask, alpha=0.5, cmap='hot')
    plt.title(f"Detected {object_name}s")
    plt.axis('off')
    
    # Count display
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f"Count: {count}\n{object_name}s", 
             fontsize=24, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title("Result")
    
    plt.tight_layout()
    plt.savefig('counting_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to 'counting_result.png'")

def main():
    parser = argparse.ArgumentParser(description='Count objects in custom images')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--object', type=str, default='flower', help='Object to count (e.g., "hibiscus flower", "red flower")')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--threshold', type=float, default=0.25, help='Similarity threshold')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return
    
    # Check if SAM model exists
    if not os.path.exists("./pretrain/sam_vit_b_01ec64.pth"):
        print("Error: SAM model not found! Please download sam_vit_b_01ec64.pth to ./pretrain/")
        return
    
    try:
        count, masks = count_objects_in_image(args.image, args.object, args.device)
        print(f"\nFinal result: Found {count} {args.object}(s) in the image!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()