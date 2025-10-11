import torch
from pathlib import Path
import json
import cv2
import argparse
from train_e2e import E2EModel
from tqdm import tqdm  # For progress bar

def resize_frame_for_display(frame, max_width=800, max_height=600):
    """
    Resize the frame to fit within the max_width and max_height while maintaining the aspect ratio.
    """
    height, width = frame.shape[:2]
    
    # Calculate the scaling factor to fit within max_width and max_height
    scale_x = max_width / width
    scale_y = max_height / height
    scale = min(scale_x, scale_y)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame

def annotate_frame(frame, prediction, probability):
    """
    Annotate the given frame with prediction and probability.
    """
    # Assuming prediction and probability are numpy arrays or tensors, extract the scalar values
    prediction_value = prediction[0][0]
    probability_value = max(probability[0][0])
    
    text = f"Prediction: {prediction_value}, Probability: {probability_value:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def main(video_path, output_path):
    # Define the path to the checkpoint and configuration files
    checkpoint_dir = Path(r'C:\Users\ULTRAPC\Desktop\Resultats\ReGnet-Y_clip_en_16')
    checkpoint_file = checkpoint_dir / "optim_041.pt"
    config_file = checkpoint_dir / "config.json"

    # Load the configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract parameters from the config file
    batch_size = config["batch_size"]
    clip_len = config["clip_len"]
    crop_dim = config["crop_dim"]
    num_classes = config["num_classes"]
    feature_arch = config["feature_arch"]
    temporal_arch = config["temporal_arch"]
    modality = config["modality"]

    # Initialize the model based on the configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = E2EModel(
        num_classes=num_classes,
        feature_arch=feature_arch,
        temporal_arch=temporal_arch,
        clip_len=clip_len,
        modality=modality,
        device=device
    )

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model._model.load_state_dict(checkpoint, strict=False)

    # Set the model to evaluation mode
    model._model.eval()

    # Open the video file
    #print(f"\n\nvideo_path => {video_path}\n\n")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #print(f"\n\nfps => {fps}")
    #print(f"frame_width => {frame_width}")
    #print(f"frame_height => {frame_height}\n\n")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    #print(f"\n\fourcc => {fourcc}")
    #print(f"out => {out}\n\n")
    
    # Total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for the model
        resized_frame = cv2.resize(frame, (crop_dim, crop_dim))
        input_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Shape: (1, 3, H, W)

        # Perform inference
        with torch.no_grad():
            predictions, probabilities = model.predict(input_tensor)
        
        #print(f"\n\predictions => {predictions}")
        #print(f"probabilities => {probabilities}\n\n")

        # Annotate frame
        annotated_frame = annotate_frame(frame, predictions, probabilities)

        # Resize annotated frame for display
        display_frame = resize_frame_for_display(annotated_frame, max_width=1000, max_height=800)

        # Show the resized frame in a window
        cv2.imshow("Annotated Frame", display_frame)

        # Write annotated frame to output video
        out.write(annotated_frame)

        # Update the progress bar
        progress_bar.update(1)

        # Check if 'q' is pressed to quit the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    progress_bar.close()

    print(f"Processed video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with a model and annotate results.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_path", type=str, help="Path to save the annotated output video")
    args = parser.parse_args()

    main(args.video_path, args.output_path)
