import glob
import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_PATH = r'N:\DATA-CODE' 
CLASS_TXT_FILE_PATH = "./class.txt"

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)

def save_classes_txt(classes: set, file_path: str):
    with open(file_path, "w+") as f:
        f.write("\n".join(sorted(classes)))

def get_classes(base_path):
    path = os.path.join(base_path, 'labels', '**', '*')
    games_labels_path = glob.glob(path, recursive=True)

    classes = set()
    for game_labels_path in games_labels_path:
        labels_ball_json = load_json(game_labels_path)
        annotations = labels_ball_json.get("annotations", [])
        for annotation in annotations:
            classes.add(annotation.get("label", "Unknown"))
    return classes

def time_to_seconds(time_txt: str):
    hours, minutes, seconds = map(int, time_txt.split(":"))
    return hours * 3600 + minutes * 60 + seconds

def get_full_dataset(base_path):
    fps = 30
    path = os.path.join(base_path, 'labels', '*', '*', "*")
    # print(path)
    games_labels_path = glob.glob(path)

    data_dict = {
        "video": [],
        "num_frames": [],
        "num_events": [],
        "event-frame": [],
        "event-label": [],
        "event-team": [],
        "event-position": [],
        "event-visibility": [],
    }

    for game_labels_path in tqdm(games_labels_path, total=len(games_labels_path), desc="Processing Games"):
        # print(games_labels_path)
        labels_data = load_json(game_labels_path)
        annotations = labels_data.get("annotations", [])
        
        video = game_labels_path.split(os.sep)[-3]
        num_frames = len(glob.glob(os.path.join(base_path, "frames", video, "**"), recursive=True))

        for annotation in annotations:
            frame_number = time_to_seconds(annotation["gameTime"]) * fps
            data_dict["video"].append(video)
            data_dict["num_frames"].append(num_frames)
            data_dict["num_events"].append(len(annotations))
            data_dict["event-frame"].append(frame_number)
            data_dict["event-label"].append(annotation.get("label", "Unknown"))
            data_dict["event-team"].append(annotation.get("team", "Unknown"))
            data_dict["event-position"].append(annotation.get("position", {}))
            data_dict["event-visibility"].append(annotation.get("visibility", "Unknown"))

    return pd.DataFrame(data_dict)

def split_and_save_dataset(df, output_dir, train_ratio=0.7, val_ratio=0.15):
    os.makedirs(output_dir, exist_ok=True)

    # Split 
    train, temp = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
    val, test = train_test_split(temp, test_size=val_ratio / (1 - train_ratio), random_state=42)

    # Save datasets as JSON
    for split_name, split_df in zip(["train", "val", "test"], [train, val, test]):
        split_path = os.path.join(output_dir, f"{split_name}.json")
        formatted_data = []

        for video, group in split_df.groupby("video"):
            events = [
                {
                    "frame": int(row["event-frame"]),
                    "label": row["event-label"],
                    "team": row["event-team"],
                    "position": row["event-position"],
                    "visibility": row["event-visibility"],
                }
                for _, row in group.iterrows()
            ]
            formatted_data.append({
                "video": video,
                "num_frames": int(group["num_frames"].iloc[0]),
                "num_events": int(group["num_events"].iloc[0]),
                "events": events
            })

        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=4)

    print(f"Datasets saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    # print("Classes Generating:")
    # classes = get_classes(BASE_PATH)
    # save_classes_txt(classes, "./class.txt")

    df = get_full_dataset(BASE_PATH)
    print("Dataset Loaded:")
    print(df.head())

    # Save the dataset splits
    split_and_save_dataset(df, output_dir=".")
