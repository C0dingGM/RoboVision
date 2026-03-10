import cv2
from ultralytics import YOLO
import numpy as np

class ChessDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize chess piece detector
        Args:
            model_path: Path to YOLO model (use pretrained or custom trained)
        """
        self.model = YOLO(model_path)
        
    def detect_pieces(self, frame):
        """
        Detect chess pieces in frame
        Returns: annotated frame and detections
        """
        results = self.model(frame)
        
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
        
        return frame, detections

    def run_camera(self, camera_id=0):
        """
        Run real-time detection from camera
        Args:
            camera_id: Camera device ID (0 for default)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            annotated_frame, detections = self.detect_pieces(frame)
            
            cv2.imshow('Chess Piece Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ChessDetector()
    detector.run_camera()
