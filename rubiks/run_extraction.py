from read_faces import batch_extract_faces

image_paths = ["test_cube1.jpg", "test_cube2.jpg"]
results = batch_extract_faces(image_paths, save_visuals=True, auto_open=True)

for img_result in results:
    print(f"\nImage: {img_result['image']}")
    for i, face in enumerate(img_result["faces"]):
        print(f" Face {i+1}: {face['face']}")
        print(f"  Confidences: {face['confidences']}")