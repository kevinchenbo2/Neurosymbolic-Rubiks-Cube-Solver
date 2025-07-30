import cv2 as cv
import numpy as np

def detect_face_like_quadrilaterals(image: np.ndarray, min_area: int = 1000, 
                                   canny_low: int = 50, canny_high: int = 200,
                                   debug: bool = False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, canny_low, canny_high)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"Found {len(contours)} total contours")
    
    quads = []
    for i, cnt in enumerate(contours):
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        area = cv.contourArea(approx)
        is_convex = cv.isContourConvex(approx)
        
        if debug:
            print(f"Contour {i}: {len(approx)} vertices, area={area:.1f}, convex={is_convex}")
        
        if len(approx) == 4 and area > min_area and is_convex:
            quads.append(approx)
            if debug:
                print(f"  -> Accepted as quad!")
    
    if debug:
        print(f"Final quads found: {len(quads)}")
    
    return sorted(quads, key=cv.contourArea, reverse=True)[:3]

def test_thresholds(image_path, canny_low=30, canny_high=100, min_area=500):
    print(f"\nTesting with canny_low={canny_low}, canny_high={canny_high}, min_area={min_area}")
    
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    quads = detect_face_like_quadrilaterals(
        image, 
        min_area=min_area, 
        canny_low=canny_low, 
        canny_high=canny_high,
        debug=True
    )
    
    print(f"Found {len(quads)} quadrilaterals")
    return quads

# Test different combinations
image_path = "test_cube1.jpg"

# Test 1: Lower Canny thresholds
test_thresholds(image_path, canny_low=30, canny_high=100, min_area=1000)

# Test 2: Even lower Canny thresholds
test_thresholds(image_path, canny_low=20, canny_high=80, min_area=1000)

# Test 3: Lower minimum area
test_thresholds(image_path, canny_low=30, canny_high=100, min_area=500)

# Test 4: Very permissive settings
test_thresholds(image_path, canny_low=10, canny_high=50, min_area=200)