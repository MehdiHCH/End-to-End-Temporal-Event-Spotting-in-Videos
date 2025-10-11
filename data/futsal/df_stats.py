import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to process JSON data into a DataFrame
def process_json_to_dataframe(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    events = [
        {
            'video': item['video'],
            'num_frames': item['num_frames'],
            'num_events': item['num_events'],
            'frame': event['frame'],
            'label': event['label'],
            'team': event['team'],
            'position': event['position'],
            'visibility': event['visibility']
        }
        for item in data
        for event in item.get('events', [])
    ]
    
    return pd.DataFrame(events)

# Process each JSON file into a DataFrame
train_data = process_json_to_dataframe('train.json')
val_data = process_json_to_dataframe('val.json')
test_data = process_json_to_dataframe('test.json')

# Combine all data for comprehensive EDA
all_data = pd.concat([train_data, val_data, test_data], keys=["Train", "Validation", "Test"])

# Print normalized label value counts
print(train_data["label"].value_counts(normalize=True))
print(val_data["label"].value_counts(normalize=True))
print(test_data["label"].value_counts(normalize=True))

# Visualization 1: Label Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=all_data.reset_index(), x="label", hue="level_0")
plt.title("Label Distribution Across Train, Validation, and Test")
plt.xlabel("Label")
plt.ylabel("Count")
plt.legend(title="Dataset")
plt.show()

# Visualization 2: Team-wise Event Counts
plt.figure(figsize=(10, 6))
sns.countplot(data=all_data, x="team", hue="label", palette="Set2")
plt.title("Event Counts by Team and Label")
plt.xlabel("Team")
plt.ylabel("Count")
plt.legend(title="Label")
plt.show()