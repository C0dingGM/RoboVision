import cv2
from ultralytics import YOLO
import numpy as np

class ChessDetector:
    def __init__(self, model_path='runs/detect/chess_pieces_14/weights/best.pt'):
        """
        Initialize chess piece detector
        Args:
            model_path: Path to YOLO model (use pretrained or custom trained)
        """
        self.model = YOLO(model_path)
        self.show_grid = False
        
    def detect_pieces(self, frame, conf_threshold=0.25):
        """
        Detect chess pieces in frame
        Returns: annotated frame and detections
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                class_name = self.model.names[cls]
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf),
                    'class': cls,
                    'name': class_name
                })
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add detection count to frame
        cv2.putText(frame, f"Detected: {len(detections)} pieces", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame, detections
    
    def draw_grid(self, frame, grid_size=8, x_offset=50, y_offset=50, cell_size=80):
        """
        Draw an 8x8 grid overlay on the frame
        Args:
            frame: Input frame
            grid_size: Number of rows/columns (default 8 for chessboard)
            x_offset: X position of top-left corner
            y_offset: Y position of top-left corner
            cell_size: Size of each grid cell in pixels
        """
        board_size = cell_size * grid_size
        
        # Draw vertical lines
        for i in range(grid_size + 1):
            x = x_offset + i * cell_size
            cv2.line(frame, (x, y_offset), (x, y_offset + board_size), (255, 0, 0), 2)
        
        # Draw horizontal lines
        for i in range(grid_size + 1):
            y = y_offset + i * cell_size
            cv2.line(frame, (x_offset, y), (x_offset + board_size, y), (255, 0, 0), 2)
        
        # Add chess board labels (a-h, 1-8)
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for i, file in enumerate(files):
            x = x_offset + i * cell_size + cell_size // 2 - 10
            cv2.putText(frame, file, (x, y_offset - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        for i in range(grid_size):
            y = y_offset + i * cell_size + cell_size // 2 + 5
            cv2.putText(frame, str(8 - i), (x_offset - 30, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame

    def run_camera(self, camera_id=0):
        """
        Run real-time detection from camera
        Args:
            camera_id: Camera device ID (0 for default)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            print("Possible solutions:")
            print("1. Check if camera permissions are granted in System Settings > Privacy & Security > Camera")
            print("2. Make sure no other app is using the camera")
            print("3. Try a different camera ID (e.g., camera_id=1)")
            return
        
        # Try reading a test frame
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Camera opened but cannot read frames")
            print("Try closing other applications using the camera")
            cap.release()
            return
        
        print("Press 'q' to quit")
        print("Press 'g' to toggle grid overlay")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            annotated_frame, detections = self.detect_pieces(frame, conf_threshold=0.05)
            
            # Draw grid if enabled
            if self.show_grid:
                annotated_frame = self.draw_grid(annotated_frame)
            
            # Print detections for debugging
            if detections:
                for det in detections:
                    print(f"Found: {det['name']} ({det['confidence']:.2f})")
            
            cv2.imshow('Chess Piece Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.show_grid = not self.show_grid
                print(f"Grid overlay: {'ON' if self.show_grid else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ChessDetector()
    detector.run_camera()
