import torch
import numbers
import random
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

label_to_int = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}

#######################
# Data converter helper
#######################
def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


############################
# Cholec80 Datast Wrapper
# Default: Returns Single Image
############################
class CholecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_paths,
        file_labels,
        transform=None,
        loader=pil_loader,
    ):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:, 0]
        self.transform = transform
        self.loader = loader
        self.class_names = list(label_to_int.keys())

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        #return imgs, labels_phase, index
        return imgs, labels_phase

    def __len__(self):
        return len(self.file_paths)
#
#
# ############################
# # Cholec80 Datast Wrapper
# # Returns Sequence of Images
# ############################
# class Cholec80DatasetSequence(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         file_paths,
#         file_labels,
#         sequence_length,
#         transform=None,
#         loader=pil_loader,
#     ):
#         self.file_paths = file_paths
#         self.file_labels_phase = file_labels[:, 0]
#         self.transform = transform
#         self.loader = loader
#         self.sequence_length = sequence_length
#
#     def get_single_item(self, index):
#         img_names = self.file_paths[index]
#         labels_phase = self.file_labels_phase[index]
#         imgs = self.loader(img_names)
#         if self.transform is not None:
#             imgs = self.transform(imgs)
#
#         return imgs, labels_phase, index
#
#     def __getitem__(self, index):
#         imgs_list = []
#         labels_list = []
#         for i in range(self.sequence_length):
#             img, label, _ = self.get_single_item(index + i)
#             imgs_list.append(img.unsqueeze(0))
#             labels_list.append(label)
#         imgs_list = torch.cat(imgs_list, dim=0)
#         labels_list = torch.Tensor(labels_list)
#         indices = torch.arange(index, index + self.sequence_length)
#         return imgs_list, labels_list[index], index
#         # return imgs_list, labels_list, indices
#         # return imgs_list, labels_list, index
#
#     def __len__(self):
#         return len(self.file_paths)


##############################
# Custom Random Crop Transform
# w/ Sequence Length
##############################
class RandomCrop(object):
    def __init__(self, size, sequence_length, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0
        self.sequence_length = sequence_length

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // self.sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


##############################
# Custom Random H-Flip Transform
# w/ Sequence Length
##############################
class RandomHorizontalFlip(object):
    def __init__(self, sequence_length):
        self.count = 0
        self.sequence_length = sequence_length

    def __call__(self, img):
        seed = self.count // self.sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


##############################
# Custom Random Rotation
# w/ Sequence Length
##############################
class RandomRotation(object):
    def __init__(self, degrees, sequence_length):
        self.degrees = degrees
        self.count = 0
        self.sequence_length = sequence_length

    def __call__(self, img):
        seed = self.count // self.sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees, self.degrees)
        return TF.rotate(img, angle)


##############################
# Custom Color Jitter
# w/ Sequence Length
##############################
class ColorJitter(object):
    def __init__(
        self, sequence_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0
        self.sequence_length = sequence_length

    def __call__(self, img):
        seed = self.count // self.sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_