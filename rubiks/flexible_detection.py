import cv2 as cv
import numpy as np
from typing import List

def detect_face_like_quadrilaterals_flexible(image: np.ndarray, min_area: int = 1000, 
                                           canny_low: int = 30, canny_high: int = 100) -> List[np.ndarray]:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, canny_low, canny_high)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
            
        # Try different approximation accuracies
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
            approx = cv.approxPolyDP(cnt, epsilon_factor * cv.arcLength(cnt, True), True)
            
            # Accept 4-6 vertices (more flexible)
            if 4 <= len(approx) <= 6 and cv.isContourConvex(approx):
                # If it's not exactly 4 vertices, try to simplify it
                if len(approx) > 4:
                    # Use a more aggressive approximation to get closer to 4 vertices
                    approx = cv.approxPolyDP(cnt, 0.08 * cv.arcLength(cnt, True), True)
                    if len(approx) == 4 and cv.isContourConvex(approx):
                        quads.append(approx)
                        break
                else:
                    quads.append(approx)
                    break
    
    return sorted(quads, key=cv.contourArea, reverse=True)[:3]

# Test the flexible detection
def test_flexible_detection():
    image = cv.imread("cube_side1.jpg")
    quads = detect_face_like_quadrilaterals_flexible(image, min_area=500)
    print(f"Found {len(quads)} quadrilaterals with flexible detection")
    
    # Visualize the results
    result = image.copy()
    for i, quad in enumerate(quads):
        cv.drawContours(result, [quad], -1, (0, 255, 0), 3)
        print(f"Quad {i+1}: {len(quad)} vertices, area={cv.contourArea(quad):.1f}")
    
    cv.imwrite("flexible_detection_result.png", result)
    print("Saved result to flexible_detection_result.png")

test_flexible_detection() 