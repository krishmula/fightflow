import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import av
import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline_cnn.model import BaselineCNN
from src.models.advanced_cnn.model import AdvancedCNN

# Reverse mapping from integer to string class
CLASS_MAP = {0: 'none', 1: 'jab', 2: 'cross', 3: 'hook', 4: 'uppercut'}

def extract_frame(video_path, frame_index):
    """Accurately extracts a single frame using PyAV (matches training logic)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    
    fps = float(stream.average_rate or stream.guessed_rate or 30.0)
    time_base = stream.time_base
    seconds = frame_index / fps
    target_pts = int(seconds / float(time_base))
    
    container.seek(target_pts, any_frame=False, backward=True, stream=stream)
    
    last_frame = None
    for frame in container.decode(video=0):
        if frame.pts is None: continue
        t = float(frame.pts * time_base)
        current_idx = int(round(t * fps))
        
        if current_idx > frame_index:
            break
        last_frame = frame.to_ndarray(format="rgb24")
        if current_idx == frame_index:
            break
            
    container.close()
    if last_frame is None:
        raise ValueError(f"Could not extract frame {frame_index} from {video_path}")
    return Image.fromarray(last_frame)

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single video frame.")
    parser.add_argument("--model", type=str, default='baseline', choices=['baseline', 'advanced'], help="Which model architecture to use")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to classify")
    parser.add_argument("--weights", type=str, help="Path to saved model weights. Defaults to checkpoints/<model>_cnn_best.pth")
    args = parser.parse_args()

    weights_path = args.weights if args.weights else f"checkpoints/{args.model}_cnn_best.pth"

    if not os.path.exists(args.video):
        print(f"Error: Video not found at {args.video}")
        return
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}. Have you trained the model yet?")
        return

    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading {args.model} model on {device}...")
    if args.model == 'advanced':
        model = AdvancedCNN(num_classes=5).to(device)
    else:
        model = BaselineCNN(num_classes=5).to(device)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Set to evaluation mode!

    # Only resize + normalize for inference! No data augmentation (flips/jitter).
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Extracting frame {args.frame} from {args.video}...")
    image = extract_frame(args.video, args.frame)
    input_tensor = val_transform(image).unsqueeze(0).to(device) # Add batch dimension -> [1, 3, 128, 128]

    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]

    punch_type = CLASS_MAP[predicted_idx]
    
    print("\n" + "="*30)
    print(f"🥊 PREDICTION: {punch_type.upper()}")
    print(f"🎯 CONFIDENCE: {confidence*100:.2f}%")
    print("="*30)
    print("All class probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  - {CLASS_MAP[i]:<10}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
