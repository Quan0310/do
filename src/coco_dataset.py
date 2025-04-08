import os
from torch.utils.data import Dataset
from src.dataagrs import Compose, Crop, VerticalFlip, HSVAdjust, Resize
import pickle
import copy
import cv2
import numpy as np


class COCODataset(Dataset):
    def __init__(self, root_path="data/COCO", year="2014", mode="train", image_size=448, is_training=True):
        if mode in ["train", "val"] and year in ["2014", "2015", "2017"]:
            self.image_path = os.path.join(root_path, "images", f"{mode}{year}")
            anno_path = os.path.join(root_path, "anno_pickle", f"COCO_{mode}{year}.pkl")
            id_list_path = pickle.load(open(anno_path, "rb"))
            self.id_list_path = list(id_list_path.values())

        # Giữ lại 10 class bạn chọn
        self.class_ids = [1, 3, 18, 17, 39, 62, 2, 72, 73, 24]
        self.classes = [
            "person",         # 1
            "car",            # 3
            "dog",            # 18
            "cat",            # 17
            "skateboard",     # 39
            "tv",             # 62
            "bicycle",        # 2
            "laptop",         # 72
            "mouse",          # 73
            "zebra"           # 24
        ]

        self.class_id_map = {cls_id: idx for idx, cls_id in enumerate(self.class_ids)}  # map old_id -> new index

        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.id_list_path)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.image_path, self.id_list_path[item]["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        objects = copy.deepcopy(self.id_list_path[item]["objects"])

        filtered_objects = []
        for obj in objects:
            coco_class_id = obj[4]
            if coco_class_id in self.class_id_map:
                obj[4] = self.class_id_map[coco_class_id]  # map lại chỉ số class theo thứ tự mới
                filtered_objects.append(obj)

        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])

        image, filtered_objects = transformations((image, filtered_objects))
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(filtered_objects, dtype=np.float32)
