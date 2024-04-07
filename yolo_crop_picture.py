import torch
import os

def get_image_addresses(folder_path):
    image_addresses = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_addresses.append(os.path.join(root, file))
    return image_addresses

folder_path = r"F:\data reid\pre"
image_addresses = get_image_addresses(folder_path)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
for i, image_path in enumerate(image_addresses):
    results = model(image_path)
    person_results = results.xyxy[results.xyxy[:, 5] == 0]  # Filter to keep only class 'person' (0)
    if len(person_results) > 0:
        save_dir = os.path.join(folder_path, "person_crops")
        os.makedirs(save_dir, exist_ok=True)
        for j, person_crop in enumerate(person_results.imgs):
            save_path = os.path.join(save_dir, f"person_{i}_{j}.jpg")
            person_crop.save(save_path)

print("Person crops saved successfully.")
