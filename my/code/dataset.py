import os
import random
import numpy as np
import torch
import pandas as pd

from pandas_streaming.df import train_test_apart_stratify
from albumentations import *
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# ============================================
# Default Aug
# ============================================
class BaseAugmentation:
    def __init__(self, resize, mean, std, crop, **args):
        # basic transform
        self.transform = transforms.Compose([
            # Resize(resize, Image.BILINEAR),
            CenterCrop(crop),
            ToTensor(),
            # Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

# ============================================
# Custom Transform Ex
# ============================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# ============================================
# Custom Augmentation Ex
# ============================================
class CustomAugmentation:
    def __init__(self, resize, mean, std, crop, **args):
        self.transform = transforms.Compose([
            CenterCrop(crop),
            # Resize(resize, Image.BILINEAR),
            ColorJitter(0.2, 0.2, 0.2),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2

class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")

class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, task_type='all', age_flag=0):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.task_type = task_type
        self.age_flag = age_flag

        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []

        # -- split by class and person
        self.all_labels = []
        self.indexs = []
        self.groups = []

        self.transform = None
        self.setup()
        self.calc_statistics()

        if self.task_type == 'all':
            self.num_classes = 18
        elif self.task_type == 'age' or self.task_type == 'mask':
            self.num_classes = 3
        elif self.task_type == 'gender':
            self.num_classes = 2

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.data_dir, profile)

            if not self.age_flag:
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    # -- split by class and person
                    self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))
                    self.indexs.append(cnt)
                    self.groups.append(id)
                    cnt += 1

            # only age >= 60 dataset
            else:
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    if age_label == 2:
                        for i in range(4):
                            self.image_paths.append(img_path)
                            self.mask_labels.append(mask_label)
                            self.gender_labels.append(gender_label)
                            self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] --- Calculating Statistics ---")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

            print('========= mean of dataset : ', self.mean)
            print('========= std of dataset : ', self.std)

    def set_transform(self, transform, age_flag):
        if age_flag:
            self.transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.CenterCrop((250, 200)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(5),
                transforms.RandomAffine(degrees=11, translate=(0.1, 0.1), scale=(0.8, 0.8)),
                transforms.ToTensor(),
                transforms.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178),
                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, "[Warning] --- You Must Set Transform ---"

        image = self.read_image(index)

        # each task label
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)

        # multi class label
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        # image transform
        image_transform = self.transform(image)

        # return multi vs each task
        if self.task_type == 'all':
            return image_transform, multi_class_label
        elif self.task_type == 'age':
            return image_transform, age_label
        elif self.task_type == 'mask':
            return image_transform, mask_label
        elif self.task_type == 'gender':
            return image_transform, gender_label


    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self, aug_flag=False) -> Tuple[Subset, Subset]:
        # -- transformed dataset
        if aug_flag:
            self.val_ratio = 0.8

        df = pd.DataFrame({"indexs": self.indexs, "groups": self.groups, "labels": self.all_labels})
        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)
        train_index = train["indexs"].tolist()
        valid_index = valid["indexs"].tolist()
        return [Subset(self, train_index), Subset(self, valid_index)]

# Dataset based on people
class MaskSplitByProfileDataset(MaskBaseDataset):

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, task_type='all', age_flag=0):
        self.indices = defaultdict(list)
        self.task_type = task_type
        self.age_flag = age_flag
        super().__init__(data_dir, mean, std, val_ratio, self.task_type, self.age_flag)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self, aug_flag=False) -> List[Subset]:
        if aug_flag:
            self.val_ratio = 0.5
            self.clear_all()
            self.setup()
        return [Subset(self, indices) for phase, indices in self.indices.items()]

    # -- clear all list to transformed dataset
    def clear_all(self):
        self.image_paths.clear()
        self.mask_labels.clear()
        self.gender_labels.clear()
        self.age_labels.clear()
        self.indices = defaultdict(list)

# -- split by people based on trian : val ratio
class MaskSplitByClassDataset(MaskSplitByProfileDataset):

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, task_type='all', age_flag=0):
        self.indices = defaultdict(list)
        self.task_type = task_type
        self.age_flag = age_flag
        super().__init__(data_dir, mean, std, val_ratio, task_type, age_flag)

    def split_dataset(self, aug_flag=False) -> List[Subset]:
        from sklearn.model_selection import StratifiedShuffleSplit
        # -- add transformed dataset to original train set
        if aug_flag:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=43)

        indices = list(range(len(self.image_paths)))
        train_df = pd.DataFrame({'mask': self.mask_labels, 'gender': self.gender_labels, 'age': self.age_labels})
        train_df['label'] = train_df[['mask', 'gender', 'age']].apply(
            lambda x: self.encode_multi_class(x[0], x[1], x[2]), axis=1)

        if self.task_type == 'all':
            self.task_type = 'label'

        for train_index, test_index in sss.split(indices, train_df[self.task_type]):
            print(len(train_index), len(test_index))
        print(train_df[self.task_type].iloc[train_index].value_counts().sort_index())
        print(train_df[self.task_type].iloc[test_index].value_counts().sort_index())

        return [Subset(self, train_index), Subset(self, test_index)]

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            # Resize(resize, Image.BILINEAR),
            CenterCrop((250, 200)),
            ToTensor(),
            # Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


# custom dataset for undersampling
class CustomDataset(Dataset):
    '''
    CustomDataset for applying different transform on train / val

    Args:
         data_df: A data frame from train.csv
    '''

    class_labels = []
    image_images = []

    def __init__(self, data_df, transform=None):
        self.mean = mean
        self.std = std
        self.transform = transform
        self.df = data_df
        self.setup()

    def setup(self):
        for index in tqdm.tqdm(range(self.__len__())):
            df_series = self.df.iloc[index]
            img_path = df_series[df_idx["path"]]
            if os.path.exists(img_path):
                self.image_images.append(self.transform(image=np.array(Image.open(img_path)))['image'])
                self.class_labels.append(df_series[df_idx["label"]])

    def __getitem__(self, index):
        image = self.image_images[index]
        class_label = self.class_labels[index]
        return image, class_label

    def __len__(self):
        return len(self.df)