import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os

def detect_cube_faces_with_pretrained(image_path, confidence_threshold=0.3):
    """
    Use pre-trained YOLO to detect objects, then filter for cube-like objects
    """
    # Load pre-trained YOLO model
    model = YOLO('yolov8n.pt')  # Use the smallest model for speed
    
    print(f"Running YOLO detection on {image_path}...")
    results = model(image_path)
    
    potential_cubes = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = model.names[class_id]
                
                # Calculate bounding box properties
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height
                
                print(f"Detected: {class_name} (confidence: {confidence:.2f}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                print(f"  Width: {width:.0f}, Height: {height:.0f}, Aspect ratio: {aspect_ratio:.2f}")
                
                # Filter for potential cube faces
                if is_potential_cube_face(class_name, confidence, aspect_ratio, area):
                    potential_cubes.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id,
                        'aspect_ratio': aspect_ratio,
                        'area': area
                    })
                    print(f"  -> ACCEPTED as potential cube face!")
                else:
                    print(f"  -> REJECTED")
    
    return potential_cubes

def is_potential_cube_face(class_name, confidence, aspect_ratio, area):
    """
    Filter function to determine if a detected object could be a cube face
    """
    # Confidence threshold
    if confidence < 0.3:
        return False
    
    # Look for objects that are roughly square
    if not (0.7 <= aspect_ratio <= 1.3):
        return False
    
    # Check for reasonable size (not too small, not too large)
    if area < 1000 or area > 100000:  # Adjust these thresholds based on your images
        return False
    
    # Look for specific object classes that might be cubes
    cube_like_classes = [
        'box', 'package', 'container', 'object', 'thing', 'item'
    ]
    
    # Accept if it's a cube-like class or if it meets geometric criteria
    if class_name.lower() in cube_like_classes:
        return True
    
    # Also accept high-confidence detections that are square-shaped
    if confidence > 0.6 and 0.8 <= aspect_ratio <= 1.2:
        return True
    
    return False

def convert_bbox_to_quadrilateral(bbox):
    """
    Convert bounding box [x1, y1, x2, y2] to quadrilateral format
    """
    x1, y1, x2, y2 = bbox
    quad = np.array([
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2]   # Bottom-left
    ], dtype=np.int32)
    return quad

def visualize_detections(image_path, detections, save_path=None):
    """
    Visualize the detected potential cube faces
    """
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Draw all detections
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_name}: {confidence:.2f}"
        cv.putText(image, label, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add detection number
        cv.putText(image, f"#{i+1}", (x1, y2+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        print(f"Detection #{i+1}: {class_name} at {bbox}")
    
    # Save or display
    if save_path:
        cv.imwrite(save_path, image)
        print(f"Saved visualization to: {save_path}")
    else:
        cv.imshow("Detections", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

def test_pretrained_detection():
    """
    Test the pre-trained YOLO detection on your cube images
    """
    image_paths = ["cube_side1.jpg", "cube_side2.jpg"]
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing: {image_path}")
        print(f"{'='*50}")
        
        # Run detection
        detections = detect_cube_faces_with_pretrained(image_path)
        
        print(f"\nResults for {image_path}:")
        print(f"Found {len(detections)} potential cube faces")
        
        if detections:
            # Convert to quadrilateral format for your existing code
            quads = []
            for detection in detections:
                quad = convert_bbox_to_quadrilateral(detection['bbox'])
                quads.append(quad)
                print(f"Quad: {quad.tolist()}, Confidence: {detection['confidence']:.2f}")
            
            # Visualize results
            save_path = f"yolo_detection_{os.path.splitext(image_path)[0]}.png"
            visualize_detections(image_path, detections, save_path)
            
            return quads  # Return the first set of detections for testing
        else:
            print("No potential cube faces detected")
    
    return []

def integrate_with_existing_code():
    """
    Function to integrate with your existing read_faces.py code
    """
    def detect_face_like_quadrilaterals_yolo(image: np.ndarray, min_area: int = 1000):
        """
        Replace the original detect_face_like_quadrilaterals function
        """
        # Save image temporarily
        temp_path = "temp_image.jpg"
        cv.imwrite(temp_path, image)
        
        # Run YOLO detection
        detections = detect_cube_faces_with_pretrained(temp_path)
        
        # Convert to quadrilateral format
        quads = []
        for detection in detections:
            if detection['area'] >= min_area:
                quad = convert_bbox_to_quadrilateral(detection['bbox'])
                quads.append(quad)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return quads
    
    return detect_face_like_quadrilaterals_yolo

if __name__ == "__main__":
    # Test the detection
    test_pretrained_detection() 