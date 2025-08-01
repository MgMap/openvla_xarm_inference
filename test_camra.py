# Test script - save as test_camera.py
import cv2

def check_all_cameras():
    """Check which camera indices are available."""
    print("Checking available cameras...")
    available_cameras = []
    
    # Test camera indices 0-10
    for i in range(21):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                    'frame_shape': frame.shape
                })
                print(f"✓ Camera {i}: {width}x{height} @ {fps:.1f} FPS (Backend: {backend})")
            else:
                print(f"✗ Camera {i}: Opens but cannot read frames")
            cap.release()
        else:
            print(f"✗ Camera {i}: Cannot open")
    
    return available_cameras

def test_specific_camera(camera_index):
    """Test a specific camera with live preview."""
    print(f"\nTesting camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return False
    
    # Try to set some properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Camera {camera_index} opened. Press 'q' to quit, 's' to save image, 'n' for next camera")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            
            # Add info overlay
            cv2.putText(frame, f"Camera {camera_index} - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Shape: {frame.shape}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q'=quit, 's'=save, 'n'=next camera", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow(f'Camera {camera_index} Test', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False  # Quit entirely
            elif key == ord('n'):
                cap.release()
                cv2.destroyWindow(f'Camera {camera_index} Test')
                return True   # Next camera
            elif key == ord('s'):
                filename = f'camera_{camera_index}_test.jpg'
                cv2.imwrite(filename, frame)
                print(f"Image saved as {filename}")
        else:
            print("Failed to read frame")
            break
    
    cap.release()
    cv2.destroyWindow(f'Camera {camera_index} Test')
    return True

def test_camera_simple():
    """Original simple test for camera 4."""
    cap = cv2.VideoCapture(6)  # Use same index as your script
    
    if not cap.isOpened():
        print("Cannot open camera 4")
        return
    
    print("Camera 4 opened. Press 'q' to quit, 's' to save image")
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera 4 Test', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('saved_image.jpg', frame)
                print("Image saved as saved_image.jpg")
        else:
            print("Failed to read frame")
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=== Camera Detection and Testing ===")
    
    # First, check all available cameras
    available = check_all_cameras()
    
    if not available:
        print("\n❌ No cameras found!")
        return
    
    print(f"\n✅ Found {len(available)} working camera(s):")
    for cam in available:
        print(f"   Index {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Test all cameras one by one")
    print("2. Test only camera 4 (your original)")
    print("3. Test specific camera index")
    print("4. Exit")
    
    try:
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == "1":
            print("\nTesting all cameras (press 'n' for next, 'q' to quit)...")
            for cam in available:
                if not test_specific_camera(cam['index']):
                    break  # User pressed 'q'
                    
        elif choice == "2":
            if any(cam['index'] == 4 for cam in available):
                test_camera_simple()
            else:
                print("Camera 4 is not available!")
                
        elif choice == "3":
            index = int(input("Enter camera index: "))
            if any(cam['index'] == index for cam in available):
                test_specific_camera(index)
            else:
                print(f"Camera {index} is not available!")
                
        elif choice == "4":
            print("Exiting...")
            
        else:
            print("Invalid choice!")
            
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()