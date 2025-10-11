import torch
from pathlib import Path
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from train_e2e import E2EModel
from tqdm import tqdm

VIDEOS_PATH = r"C:\\Users\\ULTRAPC\\Desktop\\DATA-CODE\\videos"

def load_test_data(json_path):
    """Load the test data from the JSON file."""
    with open(json_path, 'r') as f:
        test_data = json.load(f)
    return test_data

cls_map = {
    0: "ANY",
    1: "DRIVE",
    2: "PASS"
}

def evaluate_model(test_data, model, device, crop_dim):
    """
    Evaluate the E2EModel on the test dataset.
    Handles cases where each "video" field corresponds to two separate video files (1st and 2nd half).
    Returns predictions, ground truth, and evaluation metrics.
    """
    all_predictions = []
    all_ground_truth = []

    for sample in test_data:
        video_name = sample["video"]
        events = sample["events"]

        # Define paths for the first and second halves
        sub_video_names = os.listdir(os.path.join(VIDEOS_PATH, f"{video_name}"))
        
        for sub_video_name in sub_video_names:
            if sub_video_name.startswith("1"):
                video_1_path = os.path.join(VIDEOS_PATH, f"{video_name}", sub_video_name)
            else:
                video_2_path = os.path.join(VIDEOS_PATH, f"{video_name}", sub_video_name)

        # Open the videos
        cap1 = cv2.VideoCapture(video_1_path)
        cap2 = cv2.VideoCapture(video_2_path)

        if not cap1.isOpened():
            print(f"Error: Unable to open video file {video_1_path}")
            continue

        if not cap2.isOpened():
            print(f"Error: Unable to open video file {video_2_path}")
            continue

        # Get the total number of frames for the first half
        total_frames_1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

        for event in tqdm(events, desc=f"Evaluating {video_name}", unit="event"):
            frame_number = event["frame"]
            ground_truth_label = event["label"]

            # Determine the correct video and frame offset
            if frame_number <= total_frames_1:
                # Frame belongs to the first half
                cap = cap1
                target_frame = frame_number
            else:
                # Frame belongs to the second half
                cap = cap2
                target_frame = frame_number - total_frames_1

            # Seek to the specific frame in the chosen video
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)  # 0-based indexing
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read frame {target_frame} in {video_name}")
                continue

            # Preprocess the frame
            resized_frame = cv2.resize(frame, (crop_dim, crop_dim))
            input_tensor = (
                torch.from_numpy(resized_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
            )  # Shape: (1, 3, H, W)

            # Perform inference
            with torch.no_grad():
                predictions, _ = model.predict(input_tensor)

            # Append results
            all_predictions.append(cls_map[predictions[0][0]])
            all_ground_truth.append(ground_truth_label)

        # Release the resources for the first and second halves
        cap1.release()
        cap2.release()

    return all_predictions, all_ground_truth

def compute_metrics(predictions, ground_truth):
    """Compute evaluation metrics."""
    return classification_report(ground_truth, predictions, output_dict=True)

def plot_classification_report(report, output_path):
    """Plot and save the classification report."""
    categories = list(report.keys())[:-3]  # Exclude accuracy, macro avg, weighted avg
    metrics = ['precision', 'recall', 'f1-score']

    data = np.array([[report[cat][metric] for metric in metrics] for cat in categories])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="viridis")

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(metrics)), labels=metrics)
    ax.set_yticks(np.arange(len(categories)), labels=categories)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(categories)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w")

    ax.set_title("Classification Report Metrics")
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(output_path)
    plt.close()

def main(test_json_path, device):
    result_paths = [
        r"C:\\Users\\ULTRAPC\\Desktop\\Results",
        r"C:\\Users\\ULTRAPC\\Desktop\\Results1",
        r"C:\\Users\\ULTRAPC\\Desktop\\Results2",
    ]

    for result_path in result_paths:
        # Find the .pt file dynamically
        checkpoint_files = [f for f in os.listdir(result_path) if f.startswith("optim") and f.endswith(".pt")]
        if not checkpoint_files:
            print(f"No checkpoint file found in {result_path}")
            continue

        checkpoint_path = os.path.join(result_path, checkpoint_files[0])
        config_path = os.path.join(result_path, "config.json")

        if not os.path.exists(config_path):
            print(f"Config file not found in {result_path}")
            continue

        # Load test data
        test_data = load_test_data(test_json_path)

        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Initialize the model
        model = E2EModel(
            num_classes=config["num_classes"],
            feature_arch=config["feature_arch"],
            temporal_arch=config["temporal_arch"],
            clip_len=config["clip_len"],
            modality=config["modality"],
            device=device
        )

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model._model.load_state_dict(checkpoint, strict=False)

        # Set the model to evaluation mode
        model._model.eval()

        # Evaluate the model
        predictions, ground_truth = evaluate_model(
            test_data, model, device, config["crop_dim"]
        )

        # Compute metrics
        metrics = compute_metrics(predictions, ground_truth)

        # Save and visualize metrics
        eval_path = os.path.join(result_path, "evaluation")
        os.makedirs(eval_path, exist_ok=True)

        metrics_path = os.path.join(eval_path, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        plot_path = os.path.join(eval_path, "classification_report.png")
        plot_classification_report(metrics, plot_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the E2EModel on a test dataset.")
    parser.add_argument("--test_json_path", type=str, default="./data/futsal/test.json", help="Path to the test JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the evaluation (cpu or cuda)")
    args = parser.parse_args()

    main(args.test_json_path, args.device)
