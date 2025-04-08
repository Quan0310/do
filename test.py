import json

with open("data/coco/annotations/instances_train2014_filtered.json", "r") as f:
    data = json.load(f)

categories = data['categories']
print("Tổng số nhãn:", len(categories))
print("Danh sách nhãn:", [cat['name'] for cat in categories])