import argparse
import os
import subprocess
import time
import json
from tabulate import tabulate

def run_model(script_name, video_path):
    """
    Run a specific model script with the provided video path.
    """
    print(f"Running {script_name}...")
    start_time = time.time()
    
    # Run the script as a subprocess
    cmd = ["python", script_name, "--video", video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if the script executed successfully
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        return None
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    # Extract metrics from the output
    metrics = {}
    for line in result.stdout.split('\n'):
        if "Average FPS:" in line:
            metrics['avg_fps'] = float(line.split(': ')[1].strip())
        elif "Total processing time:" in line:
            metrics['total_time'] = float(line.split(': ')[1].split(' ')[0].strip())
    
    # Add execution time
    metrics['execution_time'] = execution_time
    
    return metrics

def load_metrics_from_file(file_path):
    """
    Load metrics from a text file.
    """
    metrics = {}
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if "Average FPS:" in line:
                    metrics['avg_fps'] = float(line.split(': ')[1].strip())
                elif "Total processing time:" in line:
                    metrics['total_time'] = float(line.split(': ')[1].split(' ')[0].strip())
    
    return metrics

def generate_report(results):
    """
    Generate a comparison report of the models.
    """
    # Create table for comparison
    table = []
    for model_name, metrics in results.items():
        if metrics:
            table.append([
                model_name,
                f"{metrics.get('avg_fps', 'N/A'):.2f}" if isinstance(metrics.get('avg_fps'), (int, float)) else 'N/A',
                f"{metrics.get('total_time', 'N/A'):.2f}" if isinstance(metrics.get('total_time'), (int, float)) else 'N/A',
                f"{metrics.get('execution_time', 'N/A'):.2f}" if isinstance(metrics.get('execution_time'), (int, float)) else 'N/A'
            ])
        else:
            table.append([model_name, 'Failed', 'Failed', 'Failed'])
    
    # Print table
    headers = ["Model", "Average FPS", "Processing Time (s)", "Execution Time (s)"]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))
    
    # Save results to file
    with open('results/comparison_results.txt', 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Save results to JSON for easier programmatic access
    json_results = {}
    for model_name, metrics in results.items():
        if metrics:
            json_results[model_name] = {
                'avg_fps': metrics.get('avg_fps', 'N/A'),
                'total_time': metrics.get('total_time', 'N/A'),
                'execution_time': metrics.get('execution_time', 'N/A')
            }
        else:
            json_results[model_name] = {
                'avg_fps': 'Failed',
                'total_time': 'Failed',
                'execution_time': 'Failed'
            }
    
    with open('results/comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Run all YOLO models and compare their performance')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--skip-existing', action='store_true', help='Skip models that already have output files')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Make results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Define models to run
    models = {
        'YOLOv10n': 'track_yolov10n.py',
        'YOLOv11n': 'track_yolov11n.py',
        'YOLOv12n': 'track_yolov12n.py'
    }
    
    # Run each model and collect results
    results = {}
    
    for model_name, script_name in models.items():
        output_path = f"results/{model_name.lower()}_output.mp4"
        metrics_path = output_path.replace('.mp4', '_metrics.txt')
        
        # Check if output already exists and we're skipping existing outputs
        if args.skip_existing and os.path.exists(output_path) and os.path.exists(metrics_path):
            print(f"Output for {model_name} already exists. Loading metrics...")
            metrics = load_metrics_from_file(metrics_path)
            results[model_name] = metrics
        else:
            # Run the model
            metrics = run_model(script_name, args.video)
            results[model_name] = metrics
    
    # Generate report
    generate_report(results)

if __name__ == "__main__":
    main()
