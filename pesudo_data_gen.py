import os
import random
import json

# Directory containing training images
train_dir = "./val_data"

# Output JSON file
output_file = "val_clinical.json"

# Parameters
num_features = 5  
max_value = 100   

img_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

data_dict = {}
for img_name in img_files:
    key = os.path.splitext(img_name)[0]
    value = [str(random.randint(0, max_value)) for _ in range(num_features)]
    data_dict[key] = value

with open(output_file, "w") as f:
    json.dump(data_dict, f, indent=2)