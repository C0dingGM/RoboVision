#!/usr/bin/env python3
"""
Extract frames from videos for labeling.
"""
import cv2
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, fps=2):
    """Extract frames from video at specified fps."""
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_path = output_dir / f"{video_name}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"✓ Extracted {saved_count} frames from {video_name}")
    return saved_count

def main():
    # Setup paths
    source_dir = Path("yolo-3:15/chess_images")
    output_dir = Path("dataset/images/unlabeled")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎬 Starting frame extraction from videos...\n")
    
    # Process only .MOV videos
    videos = [f for f in source_dir.iterdir() if f.suffix == '.MOV']
    
    total_frames = 0
    for video in videos:
        frames = extract_frames_from_video(video, output_dir, fps=0.5)
        total_frames += frames
    
    print(f"\n✅ Complete! Extracted {total_frames} frames from {len(videos)} videos")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"\n💡 Next step: Run 'labelImg {output_dir.absolute()}' to start labeling")

if __name__ == "__main__":
    main()
