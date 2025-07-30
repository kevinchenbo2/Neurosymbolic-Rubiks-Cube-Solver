import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def debug_image_processing(image_path):
    print(f"\n=== Debugging {image_path} ===")
    
    # Load image
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(f"Grayscale range: {gray.min()} to {gray.max()}")
    
    # Apply Gaussian blur
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Test multiple Canny thresholds
    canny_configs = [
        (10, 30),
        (20, 60), 
        (30, 90),
        (50, 150),
        (100, 200)
    ]
    
    for canny_low, canny_high in canny_configs:
        print(f"\n--- Canny thresholds: {canny_low}, {canny_high} ---")
        
        edges = cv.Canny(blurred, canny_low, canny_high)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        # Analyze contours
        valid_quads = 0
        for i, cnt in enumerate(contours[:10]):  # Look at first 10 contours
            area = cv.contourArea(cnt)
            approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
            is_convex = cv.isContourConvex(approx)
            
            print(f"  Contour {i}: {len(approx)} vertices, area={area:.1f}, convex={is_convex}")
            
            if len(approx) == 4 and area > 500 and is_convex:
                valid_quads += 1
                print(f"    -> VALID QUAD!")
        
        print(f"Valid quads found: {valid_quads}")
        
        # Save edge image for visual inspection
        edge_filename = f"debug_edges_{canny_low}_{canny_high}.png"
        cv.imwrite(edge_filename, edges)
        print(f"Saved edge image: {edge_filename}")
    
    # Also save the original image and grayscale for comparison
    cv.imwrite("debug_original.png", image)
    cv.imwrite("debug_gray.png", gray)
    cv.imwrite("debug_blurred.png", blurred)
    print("\nSaved debug images: debug_original.png, debug_gray.png, debug_blurred.png")

# Test both images
debug_image_processing("cube_side1.jpg")
debug_image_processing("cube_side2.jpg")