import cv2 as cv
import numpy as np
import os
from typing import List, Tuple, Dict
import subprocess


# Reference Rubik’s cube colors in BGR
CUBE_COLORS_BGR = {
    'W': np.array([255, 255, 255]),
    'Y': np.array([0, 255, 255]),
    'R': np.array([0, 0, 255]),
    'O': np.array([0, 165, 255]),
    'B': np.array([255, 0, 0]),
    'G': np.array([0, 255, 0])
}


def classify_color_with_confidence(rgb: Tuple[int, int, int]) -> Tuple[str, float]:
    bgr = np.array(rgb[::-1])
    closest, min_dist = 'X', float('inf')
    for color, ref_bgr in CUBE_COLORS_BGR.items():
        dist = np.linalg.norm(bgr - ref_bgr)
        if dist < min_dist:
            closest, min_dist = color, dist
    max_dist = np.linalg.norm(np.array([255, 255, 255]))
    confidence = 1 - (min_dist / max_dist)
    return closest, round(confidence, 3)


def detect_face_like_quadrilaterals(image: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        if len(approx) == 4 and cv.contourArea(approx) > min_area and cv.isContourConvex(approx):
            quads.append(approx)
    return sorted(quads, key=cv.contourArea, reverse=True)[:3]


def warp_perspective_to_square(image: np.ndarray, quad: np.ndarray, output_size: int = 300) -> np.ndarray:
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


def extract_3x3_grid_centers(square_img: np.ndarray, grid_size: int = 3) -> List[Tuple[int, int]]:
    h, w = square_img.shape[:2]
    step_x, step_y = w // grid_size, h // grid_size
    centers = []
    for r in range(grid_size):
        for c in range(grid_size):
            x = (c * step_x) + step_x // 2
            y = (r * step_y) + step_y // 2
            centers.append((x, y))
    return centers


def extract_face_colors(warped: np.ndarray, centers: List[Tuple[int, int]]) -> Tuple[str, List[float]]:
    face_str = ""
    confidences = []
    for x, y in centers:
        roi = warped[max(y - 3, 0):y + 3, max(x - 3, 0):x + 3]
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        rgb = tuple(map(int, avg_color[::-1]))
        color, conf = classify_color_with_confidence(rgb)
        face_str += color
        confidences.append(conf)
    return face_str, confidences


def visualize_face_grid_with_labels(
    image: np.ndarray,
    centers: List[Tuple[int, int]],
    labels: List[str],
    confidences: List[float]
) -> np.ndarray:
    annotated = image.copy()
    for idx, (x, y) in enumerate(centers):
        label = labels[idx]
        conf = confidences[idx]
        text = f"{label}:{conf:.2f}"
        cv.circle(annotated, (x, y), 6, (0, 255, 0), -1)
        cv.putText(annotated, text, (x - 15, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    return annotated


import cv2 as cv
import numpy as np
import os
import subprocess  # For opening images
from typing import List, Tuple, Dict


def batch_extract_faces(image_paths: List[str], save_visuals: bool = False, auto_open: bool = False) -> List[Dict]:
    all_results = []
    for img_path in image_paths:
        image = cv.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        quads = detect_face_like_quadrilaterals(image)
        face_outputs = []
        for idx, quad in enumerate(quads):
            warped = warp_perspective_to_square(image, quad)
            centers = extract_3x3_grid_centers(warped)
            face_str, confs = extract_face_colors(warped, centers)
            vis = visualize_face_grid_with_labels(warped, centers, list(face_str), confs)
            if save_visuals:
                fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_face{idx}.png"
                cv.imwrite(fname, vis)
                print(f"Saved visualization to: {fname}")
                if auto_open:
                    try:
                        subprocess.run(["open", fname], check=True)  # macOS specific
                    except Exception as e:
                        print(f"Could not open image {fname}: {e}")
            face_outputs.append({
                "face": face_str,
                "confidences": confs
            })
        all_results.append({
            "image": img_path,
            "faces": face_outputs
        })
    return all_results


