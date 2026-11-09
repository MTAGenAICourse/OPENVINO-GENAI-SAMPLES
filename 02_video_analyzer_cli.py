import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
import cv2
import time
import argparse
import os
from datetime import datetime
from pytube import YouTube


# Suppress OpenVINO deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                       message="The `openvino.runtime` module is deprecated")



def handle_youtube_url(youtube_url, temp_dir="temp"):
    """Download YouTube video and return the file path."""
    try:
        from pytube import YouTube
        
        print(f"Detected YouTube URL: {youtube_url}")
        print(f"Downloading video... (this may take a moment)")
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download the YouTube video
        yt = YouTube(youtube_url)
        video_title = yt.title
        print(f"Found video: '{video_title}'")
        
        # Get the highest resolution stream with both video and audio
        stream = yt.streams.filter(progressive=True).order_by('resolution').desc().first()
        
        if not stream:
            # Fallback to any video stream if no progressive stream found
            stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
        
        if not stream:
            print("No suitable video stream found")
            return None
        
        print(f"Downloading {stream.resolution} video...")
        output_file = stream.download(output_path=temp_dir)
        
        print(f"Downloaded YouTube video to: {output_file}")
        return output_file
        
    except ImportError:
        print("Error: pytube package is required for YouTube support.")
        print("Please install it with: pip install pytube")
        return None
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        return None

def load_model(model_path, device="GPU"):
    """Load the OpenVINO model."""
    print(f"Loading AI model from {model_path} on {device}...")
    try:
        pipeline = ov_genai.VLMPipeline(str(model_path), device)
        # Phi-3.5 ends its turn with <|end|> (token 32007), but the exported
        # generation_config.json only lists </s>: stop there, or it keeps writing.
        _cfg = pipeline.get_generation_config()
        _cfg.stop_token_ids = {32007}
        pipeline.set_generation_config(_cfg)
        print("Model loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return None

def process_image(image):
    """Process image for model input."""
    # Convert from OpenCV BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Resize large images to prevent memory issues
    max_size = 512
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        
    # Convert to OpenVINO tensor
    image_data = np.array(pil_image).reshape(1, pil_image.height, pil_image.width, 3).astype(np.uint8)
    return pil_image, ov.Tensor(image_data)

def generate_description(pipe, image_tensor, prompt_text):
    """Generate description from image using model."""
    # Add system prompt to guide the model
    system_prompt = "You are a helpful video analysis assistant. Give direct, concise answers about what you see in this video frame. Focus only on what's visible in the current frame."
    
    # Combine system prompt and user prompt
    template = "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"
    formatted_prompt = template.format(system_prompt, prompt_text)
   
    try:
        start_time = time.time()
        result = str(pipe.generate(formatted_prompt, image=image_tensor, max_new_tokens=100))

        # Clean up response
        result = result.split("How to")[0].strip()
        result = result.split("Assistant:")[0].strip()
        result = result.split("User:")[0].strip()
        result = result.split("#")[0].strip()
        generation_time = time.time() - start_time
        return result, generation_time
    except Exception as e:
        print(f"Error analyzing frame: {str(e)}")
        return None, 0

def try_open_video_source(source):
    """Try to open video source with error handling and YouTube support."""
    # Check if source is a YouTube URL
    if "youtube.com/" in source or "youtu.be/" in source:
        source = handle_youtube_url(source)
        if source is None:
            return None
    
    # Try to open as camera index if it's a digit
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    # Check if source is a file path and if it exists
    elif isinstance(source, str) and not source.startswith(('rtsp://', 'http://', 'https://')):
        if not os.path.exists(source):
            print(f"Error: File '{source}' does not exist.")
            print("Please check the file path and try again.")
            return None
        
    print(f"Attempting to open video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        # Provide guidance based on the type of source
        if isinstance(source, int):
            print(f"Make sure camera {source} is connected and available.")
        elif isinstance(source, str):
            if source.startswith(('rtsp://', 'http://', 'https://')):
                print("Check that the stream URL is correct and the stream is active.")
            else:
                print("Make sure the file is a valid video format and is not corrupted.")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video opened successfully: {width}x{height} at {fps:.2f} FPS")
    
    return cap




def save_to_log(log_file, frame_number, timestamp, prompt, description):
    """Save analysis results to log file."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"==== Frame {frame_number} - {timestamp} ====\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Analysis: {description}\n\n")

def save_frame(frame, output_dir, frame_number):
    """Save frame as image file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"frame_{frame_number:05d}.jpg")
    cv2.imwrite(filename, frame)
    return filename

def main():
    parser = argparse.ArgumentParser(description="Video Stream Analysis CLI")
    parser.add_argument("--source", required=True, help="Video source (file path, camera index, or URL)")
    parser.add_argument("--model", default="./Phi-3.5-vision-instruct-int4-ov/", help="Path to OpenVINO model")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="Device for inference (start on CPU, then try GPU)")
    parser.add_argument("--interval", type=float, default=1.0, help="Analysis interval in seconds")
    parser.add_argument("--prompt", default="What's happening in this frame?", help="Analysis prompt")
    parser.add_argument("--display", action="store_true", help="Display video frames")
    parser.add_argument("--save-frames", action="store_true", help="Save analyzed frames")
    parser.add_argument("--output-dir", default="output", help="Directory to save frames and logs")
    args = parser.parse_args()

    # Load model
    pipe = load_model(args.model, args.device)
    if pipe is None:
        return

    # Open video source
    cap = try_open_video_source(args.source)
    if cap is None:
        return
    
    # Create output directory and log file
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"analysis_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Print info
    print("\nVideo Stream Analysis Starting...")
    print(f"Analysis interval: {args.interval} seconds")
    print(f"Analysis prompt: '{args.prompt}'")
    print(f"Log file: {log_file}")
    if args.save_frames:
        frames_dir = os.path.join(args.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
    print("\nPress 'Q' to quit\n")
    
    # Initialize variables
    frame_count = 0
    last_analysis_time = None

    # For video files, measure the interval in *video* time (frame index / fps).
    # Wall-clock time would analyze every frame back to back, because each
    # analysis takes longer than the interval. Live cameras keep wall-clock time.
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    is_file = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 and fps > 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached")
                break

            frame_count += 1
            current_time = frame_count / fps if is_file else time.time()
            
            # Display frame if requested
            if args.display:
                cv2.imshow("Video Stream", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested exit")
                    break
            
            # Check if it's time to analyze a frame
            if last_analysis_time is None or current_time - last_analysis_time >= args.interval:
                print(f"\nAnalyzing frame {frame_count}...")
                
                # Process image
                pil_image, image_tensor = process_image(frame)
                
                # Generate description
                result, generation_time = generate_description(pipe, image_tensor, args.prompt)
                
                if result:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Print results
                    print(f"Time: {timestamp}")
                    print(f"Prompt: {args.prompt}")
                    print(f"Analysis: {result}")
                    print(f"Processing time: {generation_time:.2f} seconds")
                    
                    # Save to log
                    save_to_log(log_file, frame_count, timestamp, args.prompt, result)
                    
                    # Save frame if requested
                    if args.save_frames:
                        frame_file = save_frame(frame, os.path.join(args.output_dir, "frames"), frame_count)
                        print(f"Frame saved to: {frame_file}")
                
                # Update time of last analysis
                last_analysis_time = current_time
                
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        # Clean up
        if cap is not None:
            cap.release()
        if args.display:
            cv2.destroyAllWindows()
        print("\nVideo Stream Analysis Complete")

if __name__ == "__main__":
    main()


# Basic Examples:
# Analyze a video file at 1 frame per second:
# python video_analyzer_cli.py --source video.mp4 --interval 1.0
# Analyze a webcam with custom prompt and display the video:
# python video_analyzer_cli.py --source 0 --prompt "Are there any people in this frame?" --display
# Analyze an RTSP stream, save frames, and use CPU for inference:
# python video_analyzer_cli.py --source "rtsp://your-stream-url" --device CPU --save-frames
# Analyze a YouTube video:
# python video_analyzer_cli.py --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --display
# Command Line Arguments:
# --source: Video source (file path, camera index, URL, or YouTube URL) [required]
    # --source: Video source (file path, camera index, or URL) [required]
    # --model: Path to OpenVINO model [default: "./Phi-3.5-vision-instruct-int4-ov/"]
    # --device: Device for inference (GPU or CPU) [default: GPU]
    # --interval: Analysis interval in seconds [default: 1.0]
    # --prompt: Analysis prompt [default: "What's happening in this frame?"]
    # --display: Display video frames [optional]
    # --save-frames: Save analyzed frames [optional]
    # --output-dir: Directory to save frames and logs [default: "output"]
    # python video_analyzer_cli.py --source video.mp4 --interval 1.0 --model ./Phi-3.5-vision-instruct-int4-ov/ --device GPU --display --save-frames --output-dir output


