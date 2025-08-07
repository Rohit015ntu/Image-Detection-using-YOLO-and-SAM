#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import argparse
import os
from torchvision.ops import box_iou

from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from shi_segment_anything import sam_model_registry
import clip
from utils import *

def load_models(device='cuda:0'):
    print("Loading models...")
    sam_checkpoint = "./pretrain/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return mask_generator, clip_model, preprocess

def get_bbox_from_mask(mask):
    """Extract bounding box from mask"""
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    return [int(min(x_indices)), int(min(y_indices)), int(max(x_indices)), int(max(y_indices))]

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def remove_overlapping_masks(masks, scores, iou_threshold=0.5):
    """Remove overlapping masks, keeping the ones with higher scores"""
    if len(masks) <= 1:
        return masks, scores
    
    # Calculate bounding boxes
    bboxes = []
    for mask_data in masks:
        bbox = get_bbox_from_mask(mask_data['segmentation'])
        if bbox is None:
            bboxes.append([0, 0, 1, 1])  # dummy box
        else:
            bboxes.append(bbox)
    
    # Sort by scores (descending)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    keep_indices = []
    for i in sorted_indices:
        should_keep = True
        for j in keep_indices:
            iou = calculate_iou(bboxes[i], bboxes[j])
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep_indices.append(i)
    
    filtered_masks = [masks[i] for i in keep_indices]
    filtered_scores = [scores[i] for i in keep_indices]
    
    return filtered_masks, filtered_scores

def count_objects_in_image(image_path, object_name="flower", device='cuda:0', threshold=0.3):
    mask_generator, clip_model, preprocess = load_models(device)

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Generating masks with SAM...")
    h, w = image_rgb.shape[:2]
    ref_prompt = [[w // 2, h // 2]]

    try:
        masks = mask_generator.generate(image_rgb, ref_prompt)
    except Exception as e:
        print(f"Ref prompt failed: {e}")
        ref_prompt = [[w // 4, h // 4, 3 * w // 4, 3 * h // 4]]
        masks = mask_generator.generate(image_rgb, ref_prompt)

    print(f"Generated {len(masks)} masks")

    # Filter out very small masks first
    min_mask_size = 1000  # Minimum pixels for a valid flower
    filtered_masks = []
    for mask_data in masks:
        mask = mask_data['segmentation']
        if np.sum(mask) >= min_mask_size:
            filtered_masks.append(mask_data)
    
    print(f"After size filtering: {len(filtered_masks)} masks")

    # Encode the object description once
    text_prompt = f"a photo of a {object_name}"
    text_tokens = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_vec = text_features[0]  # shape: [512]

    valid_masks = []
    scores = []

    print("Scoring masks with CLIP...")
    for i, mask_data in enumerate(filtered_masks):
        mask = mask_data['segmentation']
        
        # Create masked image
        masked_image = image_rgb.copy()
        masked_image[~mask] = 0

        pil_image = Image.fromarray(masked_image)
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Fix the tensor dimension issue
        image_features = image_features.squeeze(0)  # Remove batch dimension
        if len(image_features.shape) > 1:
            image_vec = image_features.mean(dim=0)  # Average if multiple features
        else:
            image_vec = image_features  # Already 1D

        similarity = torch.nn.functional.cosine_similarity(image_vec, text_vec, dim=0).item()
        print(f"Mask {i}: similarity = {similarity:.4f}")
        
        if similarity > threshold:
            valid_masks.append(mask_data)
            scores.append(similarity)
            print(f"  ✓ SELECTED (score: {similarity:.4f})")

    print(f"\nBefore overlap removal: {len(valid_masks)} masks")
    
    # Remove overlapping detections
    if len(valid_masks) > 1:
        valid_masks, scores = remove_overlapping_masks(valid_masks, scores, iou_threshold=0.3)
    
    count = len(valid_masks)
    print(f"After overlap removal: {count} {object_name}(s)")
    
    visualize_results(image_rgb, valid_masks, count, object_name, scores)
    return count, valid_masks

def visualize_results(image, masks, count, object_name, scores=None):
    """Visualize the counting results with better mask display"""
    plt.figure(figsize=(18, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image", fontsize=14)
    plt.axis('off')

    # Masks overlay with better visualization
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    
    if len(masks) > 0:
        # Use distinct colors for each mask
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            color = colors[i % len(colors)]
            
            # Create colored overlay
            overlay = np.zeros_like(image)
            color_rgb = mcolors.to_rgb(color)
            overlay[mask] = [int(c * 255) for c in color_rgb]
            
            # Blend with original image
            alpha = 0.4
            blended = image * (1 - alpha) + overlay * 255 * alpha
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Show just the mask area
            result = image.copy()
            result[mask] = blended[mask]
            
            # Draw contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and len(contour) > 2:
                    plt.plot(contour[:, 0], contour[:, 1], color=color, linewidth=3, alpha=0.8)
            
            # Add number labels
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                center_x, center_y = int(np.mean(x)), int(np.mean(y))
                score_text = f"{i+1}"
                if scores:
                    score_text += f"\n({scores[i]:.2f})"
                plt.text(center_x, center_y, score_text, color='white', fontsize=12, 
                        ha='center', va='center', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

    plt.title(f"Detected {object_name}s", fontsize=14)
    plt.axis('off')

    # Count display
    plt.subplot(1, 3, 3)
    # Color code the result based on accuracy
    if count <= 6:
        color = "lightgreen"
    elif count <= 15:
        color = "orange"  
    else:
        color = "lightcoral"
        
    plt.text(0.5, 0.5, f"Count: {count}\n{object_name}s", 
             fontsize=28, ha='center', va='center', weight='bold',
             bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title("Final Count", fontsize=14)

    plt.tight_layout()
    plt.savefig('counting_result.png', dpi=200, bbox_inches='tight')
    plt.show()

    print(f"Results saved to 'counting_result.png'")
    if scores:
        print("Individual detection scores:", [f"{s:.3f}" for s in scores])

def main():
    parser = argparse.ArgumentParser(description='Count objects in custom images')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--object', type=str, default='flower', help='Object to count')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--threshold', type=float, default=0.35, help='Similarity threshold (higher = more selective)')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return
    if not os.path.exists("./pretrain/sam_vit_b_01ec64.pth"):
        print("Error: SAM model checkpoint not found at ./pretrain/")
        return

    try:
        count, _ = count_objects_in_image(args.image, args.object, args.device, args.threshold)
        print(f"\n🌺 Final result: Found {count} {args.object}(s) in the image!")
        
        # Suggest threshold adjustments
        if count > 10:
            print(f"💡 Tip: Count seems high. Try increasing --threshold (current: {args.threshold})")
            print(f"   Example: --threshold 0.4 or --threshold 0.45")
        elif count == 0:
            print(f"💡 Tip: No objects found. Try lowering --threshold (current: {args.threshold})")  
            print(f"   Example: --threshold 0.25 or --threshold 0.3")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()