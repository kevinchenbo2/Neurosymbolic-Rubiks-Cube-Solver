import cv2 as cv
import numpy as np
from typing import List

def simple_cube_face_detection(image: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
    """
    Simple approach: find large, roughly square contours
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        
        # Approximate contour
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        
        if len(approx) == 4:
            # Check if it's roughly square
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            if 0.8 <= aspect_ratio <= 1.2:  # Very restrictive for squares
                # Check if the area is reasonable for a cube face
                if area > min_area and area < 100000:  # Upper limit
                    quads.append(approx)
    
    return sorted(quads, key=cv.contourArea, reverse=True)[:3]

# Test simple approach
def test_simple_detection():
    image = cv.imread("cube_side1.jpg")
    quads = simple_cube_face_detection(image, min_area=500)
    
    result = image.copy()
    for i, quad in enumerate(quads):
        cv.drawContours(result, [quad], -1, (0, 255, 0), 3)
        print(f"Quad {i+1}: area={cv.contourArea(quad):.1f}")
    
    cv.imwrite("simple_detection_result.png", result)
    print("Saved simple detection result")

test_simple_detection() 