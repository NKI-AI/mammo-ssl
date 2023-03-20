import json
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
from vissl.data.data_helper import get_mean_image
from torchvision.transforms import transforms

from vissl.utils.mammogram import Mammogram
from PIL import ImageOps


class MammoNKIDataset(Dataset):
    def __init__(self, cfg, path, split, dataset_name, data_source):
        super(MammoNKIDataset, self).__init__()
        assert data_source in [
            "mammograms",
        ], "data_source must be mammograms"

        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path

        if "CROP_DIMS" not in cfg["DATA"]["TRAIN"]:
            raise RuntimeError("No crop dimensions specified")

        self.crop_dims = cfg["DATA"]["TRAIN"]["CROP_DIMS"]

        self.data_init = False
        self._load_data()

    def _load_data(self):
        with open(self._path) as mfp:
            mammos = json.load(mfp)

        # mammos = mammos[:100000]
        if self.cfg["DATA"]["TRAIN"]["MLO_ONLY"]:
            mammos = [_ for _ in mammos if _["view"] == "MLO"]

        self.mammos = mammos
        self._num_samples = len(self.mammos)

    def __getitem__(self, index: int):
        mammo = self.mammos[index]

        mg = Mammogram(mammo["fn"])
        if len(mg.available_luts) == 0:
            img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)
            return img, False

        mg.set_random_lut()
        try:
            mg_pil = mg.as_pil
            width, height = mg_pil.size
            rescale_factor = mg.spacing / self.cfg["DATA"][self.split].REFERENCE_SPACING
            mg_pil = mg_pil.resize((int(rescale_factor * width), int(rescale_factor * height)))
            pixels = transforms.ToTensor()(mg_pil).float()

            if (
                pixels.shape[1] < self.crop_dims[0]
                or pixels.shape[2] < self.crop_dims[1]
            ):
                raise ValueError("Mammogram dimensions too small")
        except ValueError:
            img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)
            return img, False

        try:
            bounding_box = self._find_largest_bbox(mg_pil)
        except ValueError:
            bounding_box = [
                pixels.shape[1] // 2 - self.crop_dims[0] // 2,
                pixels.shape[2] // 2 - self.crop_dims[1] // 2,
            ]

        top = 0
        left = 0
        top, left = self.get_crop_location(pixels.shape, bounding_box)

        pixels = pixels[
            :, top : top + self.crop_dims[0], left : left + self.crop_dims[1]
        ]

        to_pil = transforms.ToPILImage()

        pixels_pil = to_pil(pixels)
        pixels_pil_rgb = ImageOps.colorize(
            pixels_pil, black=(0, 0, 0), white=(255, 255, 255)
        )

        assert list(pixels_pil_rgb.size) == self.crop_dims, f"wrong dimensions: {pixels_pil_rgb.size} - {self.crop_dims}"
        return pixels_pil_rgb, True

    def get_crop_location(self, pixels_shape, bounding_box):
        if bounding_box[2] < self.crop_dims[0]:
            top_min = max(0, bounding_box[0] + bounding_box[2] - self.crop_dims[0])
            top_max = min(pixels_shape[1] - self.crop_dims[0], bounding_box[0])
            top = random.randint(top_min, top_max)
        else:
            top = bounding_box[0] + random.randint(
                0, bounding_box[2] - self.crop_dims[0]
            )

        if bounding_box[3] < self.crop_dims[1]:
            left_min = max(0, bounding_box[1] + bounding_box[3] - self.crop_dims[1])
            left_max = min(pixels_shape[2] - self.crop_dims[1], bounding_box[1])
            left = random.randint(left_min, left_max)
        else:
            left = bounding_box[1] + random.randint(
                0, bounding_box[3] - self.crop_dims[1]
            )

        return top, left

    def _find_largest_bbox(self, pil):
        pix = np.array(pil)
        # threshold
        thresh = cv2.threshold(pix, 2, 255, cv2.THRESH_BINARY)[1]

        # get contour bounding boxes and draw on copy of input
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        bboxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Area of bbox
            area = w * h
            bboxes.append([y, x, h, w, area])

        # Find the largest bbox
        max_bbox = max(bboxes, key=lambda x: x[4])[:-1]  # Find max bbox and remove area

        return max_bbox

    def __len__(self):
        return len(self.mammos)

    def num_samples(self):
        return self._num_samples
