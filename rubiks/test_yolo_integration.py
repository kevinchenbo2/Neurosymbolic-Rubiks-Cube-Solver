import cv2 as cv
import numpy as np
from pretrained_yolo_detection import integrate_with_existing_code, test_pretrained_detection

def test_integration():
    """
    Test integrating YOLO detection with your existing code
    """
    # Get the YOLO-based detection function
    detect_face_like_quadrilaterals_yolo = integrate_with_existing_code()
    
    # Test on your images
    image_paths = ["cube_side1.jpg", "cube_side2.jpg"]
    
    for image_path in image_paths:
        print(f"\nTesting YOLO integration on {image_path}")
        
        # Load image
        image = cv.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
        
        # Run YOLO detection
        quads = detect_face_like_quadrilaterals_yolo(image, min_area=500)
        
        print(f"Found {len(quads)} quadrilaterals")
        
        # Visualize results
        result = image.copy()
        for i, quad in enumerate(quads):
            cv.drawContours(result, [quad], -1, (0, 255, 0), 3)
            area = cv.contourArea(quad)
            print(f"Quad {i+1}: area={area:.1f}")
        
        # Save result
        save_path = f"yolo_integration_{image_path}"
        cv.imwrite(save_path, result)
        print(f"Saved result to: {save_path}")

if __name__ == "__main__":
    # First test the basic YOLO detection
    print("Testing basic YOLO detection...")
    test_pretrained_detection()
    
    # Then test integration
    print("\n" + "="*50)
    print("Testing integration with existing code...")
    test_integration() 