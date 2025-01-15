import torch.utils.data as data
import cv2
from PIL import Image
import os


class patch_dataset(data.Dataset):
    """
        这里只保存patch列表，在需要读取数据的时候再读取图片文件
    """

    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform
        self.path = path
        self.class_to_idx = {'FTC': 0, 'BN': 1, 'PTC': 2}
        self.patches = []
        patients = os.listdir(self.path)
        for patient in patients:
            patient_path = os.path.join(self.path, patient)
            patches = os.listdir(patient_path)
            for patch in patches:
                if patch[0] == "n" and patient[:2] == "PTC":
                    continue
                patch_path = os.path.join(patient_path, patch)
                self.patches.append(patch_path)

    def __getitem__(self, idx):
        array = cv2.imread(self.patches[idx], cv2.IMREAD_COLOR)
        img = Image.fromarray(array)
        if self.transform is not None:
            img = self.transform(img)
        (_, file_name) = os.path.split(self.patches[idx])
        return img, self.class_to_idx[file_name.split('_')[1]]

    def __len__(self):
        return len(self.patches)


class slide_eval_dataset(data.Dataset):
    # 每个slide_dataset对应一个slide文件
    # 完成当前切片所有patch的计算之后，
    # 另外创建一个slide_dataset来验证另
    def __init__(self, path,
                 transform=None):
        self.path = path
        self.patches = []
        self.transform = transform
        self.label = path.split("\\")[-1].split('_')[0]
        self.class_to_idx = {'FTC': 0, 'BN': 1, 'PTC': 2}
        patches = os.listdir(os.path.join(self.path))
        for p in patches:
            self.patches.append(os.path.join(self.path, p))

    def __getitem__(self, idx):
        img = cv2.imread(self.patches[idx], cv2.IMREAD_COLOR)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[self.label]

    def __len__(self):
        return len(self.patches)