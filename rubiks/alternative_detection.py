import cv2 as cv
import numpy as np
from typing import List

def detect_face_like_quadrilaterals_alternative(image: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding instead of Canny
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
            
        # Use a more aggressive approximation
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        
        if len(approx) == 4 and cv.isContourConvex(approx):
            quads.append(approx)
    
    return sorted(quads, key=cv.contourArea, reverse=True)[:3]

# Test alternative detection
def test_alternative_detection():
    image = cv.imread("test_cube1.jpg")
    quads = detect_face_like_quadrilaterals_alternative(image, min_area=500)
    print(f"Found {len(quads)} quadrilaterals with alternative detection")
    
    # Visualize
    result = image.copy()
    for i, quad in enumerate(quads):
        cv.drawContours(result, [quad], -1, (0, 255, 0), 3)
        print(f"Quad {i+1}: {len(quad)} vertices, area={cv.contourArea(quad):.1f}")
    
    cv.imwrite("alternative_detection_result.png", result)
    print("Saved result to alternative_detection_result.png")

test_alternative_detection() 