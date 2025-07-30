import cv2 as cv
import numpy as np
from typing import List

def detect_cube_faces_by_grid_pattern(image: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
    """
    Detect cube faces by looking for 3x3 grid patterns
    """
    # Convert to different color spaces for better segmentation
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Create masks for different cube colors
    color_masks = create_cube_color_masks(hsv)
    
    # Combine all color masks
    combined_mask = np.zeros(gray.shape, dtype=np.uint8)
    for mask in color_masks:
        combined_mask = cv.bitwise_or(combined_mask, mask)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        
        # Approximate contour
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        
        # Check if it's roughly quadrilateral
        if 4 <= len(approx) <= 6:
            # Try to make it exactly 4 vertices
            if len(approx) > 4:
                approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
            
            if len(approx) == 4:
                # Check if it's roughly square-like
                x, y, w, h = cv.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                if 0.7 <= aspect_ratio <= 1.3:  # More restrictive for squares
                    # Check if the region contains cube colors in a grid pattern
                    if has_grid_pattern(image, approx):
                        quads.append(approx)
    
    return sorted(quads, key=cv.contourArea, reverse=True)[:3]

def create_cube_color_masks(hsv):
    """
    Create masks for Rubik's cube colors
    """
    masks = []
    
    # Define color ranges (more permissive)
    color_ranges = [
        # White
        (np.array([0, 0, 180]), np.array([180, 30, 255])),
        # Yellow  
        (np.array([15, 50, 50]), np.array([35, 255, 255])),
        # Red (lower)
        (np.array([0, 50, 50]), np.array([10, 255, 255])),
        # Red (upper)
        (np.array([170, 50, 50]), np.array([180, 255, 255])),
        # Orange
        (np.array([5, 50, 50]), np.array([25, 255, 255])),
        # Blue
        (np.array([100, 50, 50]), np.array([130, 255, 255])),
        # Green
        (np.array([35, 50, 50]), np.array([85, 255, 255]))
    ]
    
    for lower, upper in color_ranges:
        mask = cv.inRange(hsv, lower, upper)
        masks.append(mask)
    
    return masks

def has_grid_pattern(image, quad):
    """
    Check if a quadrilateral region contains a 3x3 grid pattern
    """
    # Warp the quadrilateral to a square
    warped = warp_perspective_to_square(image, quad, 300)
    
    # Convert to HSV
    hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
    
    # Check if the warped image has distinct color regions
    # by looking at color variance in different grid cells
    h, w = warped.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    
    color_variance = []
    for i in range(3):
        for j in range(3):
            # Extract cell
            cell = hsv[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            if cell.size > 0:
                # Calculate color variance in this cell
                mean_color = np.mean(cell, axis=(0, 1))
                variance = np.var(cell, axis=(0, 1))
                color_variance.append(np.mean(variance))
    
    # If we have good color variance, it's likely a cube face
    if len(color_variance) >= 6:  # At least 6 cells have some variance
        avg_variance = np.mean(color_variance)
        return avg_variance > 500  # Threshold for color variance
    
    return False

def warp_perspective_to_square(image: np.ndarray, quad: np.ndarray, output_size: int = 300) -> np.ndarray:
    """
    Warp a quadrilateral to a square
    """
    pts = quad.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (output_size, output_size))
    return warped

def debug_grid_detection(image_path):
    print(f"\n=== Grid Pattern Detection for {image_path} ===")
    
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Test the detection
    quads = detect_cube_faces_by_grid_pattern(image, min_area=500)
    print(f"Found {len(quads)} potential cube faces")
    
    # Visualize results
    result = image.copy()
    for i, quad in enumerate(quads):
        cv.drawContours(result, [quad], -1, (0, 255, 0), 3)
        area = cv.contourArea(quad)
        print(f"Quad {i+1}: {len(quad)} vertices, area={area:.1f}")
        
        # Save individual warped face
        warped = warp_perspective_to_square(image, quad, 300)
        cv.imwrite(f"debug_warped_face_{i}.png", warped)
        print(f"Saved warped face: debug_warped_face_{i}.png")
    
    cv.imwrite("debug_grid_detection_result.png", result)
    print("Saved result: debug_grid_detection_result.png")
    
    # Also save color masks for debugging
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    color_masks = create_cube_color_masks(hsv)
    
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, mask in enumerate(color_masks):
        combined_mask = cv.bitwise_or(combined_mask, mask)
        cv.imwrite(f"debug_color_mask_{i}.png", mask)
    
    cv.imwrite("debug_combined_color_mask.png", combined_mask)
    print("Saved color masks for debugging")

# Test the approach
debug_grid_detection("cube_side1.jpg")
debug_grid_detection("cube_side2.jpg") 