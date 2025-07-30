import cv2 as cv
import numpy as np
import math
from typing import List

def detect_cube_faces_by_hough_lines(image: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
    """
    Detect cube faces using Hough Line Transform to find grid lines
    """
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    
    # Use Probabilistic Hough Line Transform
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                          minLineLength=50, maxLineGap=10)
    
    if lines is None:
        print("No lines detected")
        return []
    
    print(f"Detected {len(lines)} lines")
    
    # Group lines by orientation
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle < 0:
            angle += 180
        
        if angle < 30 or angle > 150:  # Horizontal
            horizontal_lines.append(line[0])
        elif 60 < angle < 120:  # Vertical
            vertical_lines.append(line[0])
    
    print(f"Horizontal lines: {len(horizontal_lines)}")
    print(f"Vertical lines: {len(vertical_lines)}")
    
    # Reduce the number of lines to avoid too many intersections
    horizontal_lines = reduce_lines(horizontal_lines, max_lines=10)
    vertical_lines = reduce_lines(vertical_lines, max_lines=10)
    
    print(f"After reduction - Horizontal: {len(horizontal_lines)}, Vertical: {len(vertical_lines)}")
    
    # Find intersections
    intersections = find_intersections(horizontal_lines, vertical_lines, image.shape)
    print(f"Found {len(intersections)} intersections")
    
    # Find quadrilaterals
    quads = find_quadrilaterals_from_intersections(intersections, image.shape)
    
    return quads

def reduce_lines(lines, max_lines=10):
    """
    Reduce the number of lines by clustering similar lines
    """
    if len(lines) <= max_lines:
        return lines
    
    # Sort lines by their position (y-coordinate for horizontal, x-coordinate for vertical)
    if len(lines) > 0:
        # Determine if lines are horizontal or vertical based on first line
        x1, y1, x2, y2 = lines[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle < 0:
            angle += 180
        
        if angle < 30 or angle > 150:  # Horizontal lines
            lines.sort(key=lambda line: (line[1] + line[3]) / 2)  # Sort by y-coordinate
        else:  # Vertical lines
            lines.sort(key=lambda line: (line[0] + line[2]) / 2)  # Sort by x-coordinate
    
    # Take evenly spaced lines
    step = len(lines) // max_lines
    return lines[::step][:max_lines]

def find_intersections(horizontal_lines, vertical_lines, image_shape):
    """
    Find intersections between horizontal and vertical lines
    """
    intersections = []
    
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            intersection = line_intersection(h_line, v_line)
            if intersection is not None:
                x, y = intersection
                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    intersections.append((x, y))
    
    return intersections

def find_quadrilaterals_from_intersections(intersections, image_shape):
    """
    Find quadrilaterals from intersections (with reasonable limits)
    """
    quads = []
    
    if len(intersections) < 4:
        return quads
    
    # Limit the number of intersections to process
    if len(intersections) > 20:
        print(f"Too many intersections ({len(intersections)}), limiting to 20")
        intersections = intersections[:20]
    
    # Find the largest quadrilateral by area
    best_quad = None
    max_area = 0
    
    # Try combinations of 4 points (with reasonable limits)
    count = 0
    max_combinations = 1000  # Limit to prevent infinite loops
    
    for i in range(len(intersections) - 3):
        for j in range(i + 1, len(intersections) - 2):
            for k in range(j + 1, len(intersections) - 1):
                for l in range(k + 1, len(intersections)):
                    count += 1
                    if count > max_combinations:
                        print(f"Reached maximum combinations ({max_combinations})")
                        break
                    
                    points = [intersections[i], intersections[j], 
                            intersections[k], intersections[l]]
                    
                    quad = np.array(points, dtype=np.int32)
                    area = cv.contourArea(quad)
                    
                    if area > max_area and area > 1000:
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        
                        if width > 50 and height > 50:
                            aspect_ratio = width / height
                            if 0.5 <= aspect_ratio <= 2.0:
                                max_area = area
                                best_quad = quad
                
                if count > max_combinations:
                    break
            if count > max_combinations:
                break
        if count > max_combinations:
            break
    
    if best_quad is not None:
        quads.append(best_quad)
        print(f"Found quadrilateral with area: {max_area}")
    
    return quads

def line_intersection(line1, line2):
    """
    Find intersection point of two lines
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (int(x), int(y))

# Test the fixed version
def test_fixed_detection():
    image = cv.imread("test_cube1.jpg")
    if image is None:
        print("Could not read image")
        return
    
    quads = detect_cube_faces_by_hough_lines(image, min_area=500)
    print(f"Final result: Found {len(quads)} quadrilaterals")
    
    # Visualize
    result = image.copy()
    for i, quad in enumerate(quads):
        cv.drawContours(result, [quad], -1, (0, 255, 0), 3)
        print(f"Quad {i+1}: {len(quad)} vertices, area={cv.contourArea(quad):.1f}")
    
    cv.imwrite("fixed_hough_result.png", result)
    print("Saved result to fixed_hough_result.png")

test_fixed_detection()