import argparse
import time
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tabulate import tabulate
from ultralytics import YOLO

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def download_models():
    """
    Download YOLO models if they don't exist.
    Note: This is a placeholder. You may need to adjust this based on how models are actually available.
    """
    print("Checking for model files...")
    
    # These are placeholders - update with actual model sources when available
    models = {
        'yolov10n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10n.pt',
        'yolov11n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt',
        'yolov12n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov12n.pt'
    }
    
    for model_name, url in models.items():
        model_path = f'models/{model_name}.pt'
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            try:
                # This is a placeholder. In actual implementation, you would use something like:
                # torch.hub.download_url_to_file(url, model_path)
                print(f"Model download not implemented yet. Please manually download {model_name} to {model_path}")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
        else:
            print(f"{model_name} already exists.")

def process_video(model_name, video_path, conf_threshold=0.3, device='cuda'):
    """
    Process video with the specified YOLO model and track objects.
    Returns processing metrics.
    """
    # Load the model
    model_path = f'models/{model_name}.pt'
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
    
    # Create output video
    output_path = f'results/{model_name}_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize metrics
    processing_times = []
    frame_count = 0
    
    print(f"Processing with {model_name}...")
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}", end='\r')
        
        # Start timing
        start_time = time.time()
        
        # Run the model with tracking enabled
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
    
    print(f"\nFinished processing with {model_name}.")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    
    return {
        'model': model_name,
        'avg_fps': avg_fps,
        'total_time': total_processing_time,
        'model_size': os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    }

def generate_report(results):
    """
    Generate a comparison report of the models.
    """
    # Create table for comparison
    table = []
    for result in results:
        table.append([
            result['model'],
            f"{result['avg_fps']:.2f}",
            f"{result['total_time']:.2f}",
            f"{result['model_size']:.2f}"
        ])
    
    # Print table
    headers = ["Model", "Average FPS", "Processing Time (s)", "Model Size (MB)"]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))
    
    # Save results to file
    with open('results/comparison_results.txt', 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt="grid"))
        f.write("\n\nTest conditions:\n")
        f.write(f"- Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"- Torch version: {torch.__version__}\n")
        f.write(f"- CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"- CUDA version: {torch.version.cuda}\n")

def main():
    parser = argparse.ArgumentParser(description='Compare YOLO models for object tracking')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--download', action='store_true', help='Download models before processing')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detection')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Download models if requested
    if args.download:
        download_models()
    
    # Process video with each model
    results = []
    models = ['yolov10n', 'yolov11n', 'yolov12n']
    
    for model_name in models:
        try:
            result = process_video(model_name, args.video, args.conf, args.device)
            results.append(result)
        except Exception as e:
            print(f"Error processing with {model_name}: {e}")
    
    # Generate report
    if results:
        generate_report(results)
    else:
        print("No results to report. All processing attempts failed.")

if __name__ == "__main__":
    main()
