import gc

import numpy as np
import PIL.Image
from controlnet_aux.util import HWC3
import torch
from controlnet_aux import (CannyDetector, ContentShuffleDetector, HEDdetector,
                            LineartAnimeDetector, LineartDetector,
                            MidasDetector, MLSDdetector, NormalBaeDetector,
                            OpenposeDetector, PidiNetDetector,DWposeDetector)


from cv_utils import resize_image_by_height
from depth_estimator import DepthEstimator
from image_segmentor import ImageSegmentor
import time


class Preprocessor:
    MODEL_ID = 'lllyasviel/Annotators'

    def __init__(self):
        self.model = None
        self.name = ''

    def load(self, name: str) -> None:
        start_time = time.time()  
        if name == self.name:
            return
        if name == 'HED':
            self.model = HEDdetector.from_pretrained(self.MODEL_ID)
        elif name == 'Midas':
            self.model = MidasDetector.from_pretrained(self.MODEL_ID)
        elif name == 'MLSD':
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID)
        elif name == 'Openpose':
            self.model = OpenposeDetector.from_pretrained(self.MODEL_ID)
        elif name == 'DWpose':
            self.model = DWposeDetector(
    det_config="mmopenlab/yolox_l_8xb8-300e_coco.py",
    det_ckpt="mmopenlab/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    pose_config="mmopenlab/dwpose-l_384x288.py",
    pose_ckpt="mmopenlab/dw-ll_ucoco_384.pth"
)
        elif name == 'PidiNet':
            self.model = PidiNetDetector.from_pretrained(self.MODEL_ID)
        elif name == 'NormalBae':
            self.model = NormalBaeDetector.from_pretrained(self.MODEL_ID)
        elif name == 'Lineart':
            self.model = LineartDetector.from_pretrained(self.MODEL_ID)
        elif name == 'LineartAnime':
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID)
        elif name == 'Canny':
            self.model = CannyDetector()
        elif name == 'ContentShuffle':
            self.model = ContentShuffleDetector()
        elif name == 'DPT':
            self.model = DepthEstimator()
        elif name == 'UPerNet':
            self.model = ImageSegmentor()
        else:
            raise ValueError
        torch.cuda.empty_cache()
        gc.collect()
        print(f"time - load preproccessor: {time.time() - start_time}")
        self.name = name

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        if self.name == 'Canny':
            if 'detect_resolution' in kwargs:
                detect_resolution = kwargs.pop('detect_resolution')
                image = np.array(image)
                image = HWC3(image)
                image = resize_image_by_height(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            return PIL.Image.fromarray(image)
        elif self.name == 'Midas':
            detect_resolution = kwargs.pop('detect_resolution', 512)
            image_resolution = kwargs.pop('image_resolution', 512)
            image = np.array(image)
            image = HWC3(image)
            image = resize_image_by_height(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            image = HWC3(image)
            image = resize_image_by_height(image, resolution=image_resolution)
            return PIL.Image.fromarray(image)
        else:
            return self.model(image, **kwargs)
