import os
import re
import shutil
from rich.progress import track

class DatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(f'{dataset_path}/images', 'train')
        self.test_path = os.path.join(f'{dataset_path}/images', 'test')
        self.val_path = os.path.join(f'{dataset_path}/images', 'val')
        self.class_labels = []

    def create_folders(self):
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)

    def load_class_labels(self, classes_file):
        with open(classes_file, "r") as file:
            for line in file:
                label = line.strip().replace("\t", "").title()
                self.class_labels.append(label)

    def create_class_subfolders(self):
        for label in self.class_labels:
            train_class_folder = os.path.join(self.train_path, label)
            test_class_folder = os.path.join(self.test_path, label)
            val_class_folder = os.path.join(self.val_path, label)
            os.makedirs(train_class_folder, exist_ok=True)
            os.makedirs(test_class_folder, exist_ok=True)
            os.makedirs(val_class_folder, exist_ok=True)

    def move_files(self, filename):
        if filename == "val.txt":
            file_path = os.path.join(self.dataset_path, "val.txt")
            destination_path = self.val_path
        elif filename == "train.txt":
            file_path = os.path.join(self.dataset_path, "train.txt")
            destination_path = self.train_path
        elif filename == "test.txt":
            file_path = os.path.join(self.dataset_path, "test.txt")
            destination_path = self.test_path
        
        with open(file_path, "r") as file:
            for line in track(file, description="Extracting dataset ..."):
                data = line.strip().split(" ")
                if len(data) == 2:
                    image_name, class_label = data
                    class_folder = self.class_labels[int(class_label)]
                    source_file = os.path.join(self.dataset_path, "images", image_name)
                    destination_folder = os.path.join(destination_path, class_folder)
                    shutil.move(source_file, destination_folder)

