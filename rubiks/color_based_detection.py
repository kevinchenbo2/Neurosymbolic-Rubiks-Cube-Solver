import cv2 as cv
import numpy as np
from typing import List

def detect_cube_faces_by_color(image: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
    """
    Detect cube faces by looking for regions with Rubik's cube colors
    """
    # Convert to HSV for better color segmentation
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Define color ranges for Rubik's cube colors in HSV
    color_ranges = {
        'white': ([0, 0, 200], [180, 30, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'red': ([0, 100, 100], [10, 255, 255]),  # Lower red
        'red2': ([170, 100, 100], [180, 255, 255]),  # Upper red
        'orange': ([10, 100, 100], [20, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255])
    }
    
    # Create a mask for all cube colors
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv.inRange(hsv, lower, upper)
        combined_mask = cv.bitwise_or(combined_mask, mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
    
    # Find contours in the color mask
    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
            
        # Approximate the contour
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        
        # Accept 4-6 vertices and check if it's roughly square-like
        if 4 <= len(approx) <= 6:
            # Check if the shape is roughly rectangular
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.0:  # Accept reasonable aspect ratios
                quads.append(approx)
    
    return sorted(quads, key=cv.contourArea, reverse=True)[:3]

def debug_color_detection(image_path):
    print(f"\n=== Color-based detection for {image_path} ===")
    
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Test individual color masks
    color_ranges = {
        'white': ([0, 0, 200], [180, 30, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'red': ([0, 100, 100], [10, 255, 255]),
        'red2': ([170, 100, 100], [180, 255, 255]),
        'orange': ([10, 100, 100], [20, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255])
    }
    
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv.inRange(hsv, lower, upper)
        combined_mask = cv.bitwise_or(combined_mask, mask)
        
        # Save individual color masks
        cv.imwrite(f"debug_mask_{color_name}.png", mask)
        print(f"Saved {color_name} mask: debug_mask_{color_name}.png")
    
    # Save combined mask
    cv.imwrite("debug_combined_mask.png", combined_mask)
    print("Saved combined mask: debug_combined_mask.png")
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_OPEN, kernel)
    cv.imwrite("debug_cleaned_mask.png", cleaned_mask)
    print("Saved cleaned mask: debug_cleaned_mask.png")
    
    # Find contours
    contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours in color mask")
    
    # Analyze contours
    for i, cnt in enumerate(contours[:5]):
        area = cv.contourArea(cnt)
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        x, y, w, h = cv.boundingRect(approx)
        aspect_ratio = float(w) / h
        
        print(f"  Contour {i}: {len(approx)} vertices, area={area:.1f}, aspect_ratio={aspect_ratio:.2f}")
        
        if 4 <= len(approx) <= 6 and 0.5 <= aspect_ratio <= 2.0:
            print(f"    -> POTENTIAL CUBE FACE!")
    
    # Test the detection function
    quads = detect_cube_faces_by_color(image, min_area=500)
    print(f"Final quads found: {len(quads)}")
    
    # Visualize results
    result = image.copy()
    for i, quad in enumerate(quads):
        cv.drawContours(result, [quad], -1, (0, 255, 0), 3)
        print(f"Quad {i+1}: {len(quad)} vertices, area={cv.contourArea(quad):.1f}")
    
    cv.imwrite("debug_color_detection_result.png", result)
    print("Saved result: debug_color_detection_result.png")

# Test both images
debug_color_detection("test_cube1.jpg")
debug_color_detection("test_cube2.jpg") 