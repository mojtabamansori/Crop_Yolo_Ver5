import torch
import os

def get_image_addresses(folder_path):
    image_addresses = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_addresses.append(os.path.join(root, file))
    return image_addresses

folder_path = r"F:\data reid"
image_addresses = get_image_addresses(folder_path)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = 'WhatsApp Image 2024-04-03 at 19.10.47.jpeg'
results = model(image_addresses)
crops = results.crop(save=True)
