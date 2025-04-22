import argparse
import time
import os
import cv2
import torch
from ultralytics import YOLO

def track_objects(video_path, output_path, conf_threshold=0.3, device='cuda'):
    """
    Track objects in video using YOLOv11n model.
    """
    # Load YOLOv11n model
    model_path = 'models/yolov11n.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize metrics
    processing_times = []
    frame_count = 0
    
    print(f"Processing video with YOLOv11n...")
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}", end='\r')
        
        # Start timing
        start_time = time.time()
        
        # Process frame with tracking
        results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Write the frame to output video
        out.write(annotated_frame)
    
    # Clean up
    cap.release()
    out.release()
    
    # Calculate metrics
    avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
    total_processing_time = sum(processing_times)
    
    print(f"\nFinished processing with YOLOv11n.")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    
    # Save metrics to file
    metrics_path = output_path.replace('.mp4', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: YOLOv11n\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

def main():
    parser = argparse.ArgumentParser(description='Track objects with YOLOv11n')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='results/yolov11n_output.mp4', help='Path to output video')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detection')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Process video
    track_objects(args.video, args.output, args.conf, args.device)

if __name__ == "__main__":
    main()
