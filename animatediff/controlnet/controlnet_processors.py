# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from controlnet_aux import HEDdetector, OpenposeDetector, NormalBaeDetector


class CannyProcessor:
    def __init__(self, t1, t2, **kwargs):
        self.t1 = t1
        self.t2 = t2

    def __call__(self, input_image):
        image = np.array(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.Canny(image, self.t1, self.t2)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
        return control_image


class DepthProcessor:
    def __init__(self, **kwargs):
        self.depth_estimator = pipeline('depth-estimation')

    def __call__(self, input_image):
        image = self.depth_estimator(input_image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
        return control_image


CONTROLNET_PROCESSORS = {
    'canny': {
        'controlnet': 'lllyasviel/control_v11p_sd15_canny',
        'processor': CannyProcessor,
        'processor_params': {'t1': 50, 't2': 150},
        'is_custom': True,
    },
    'depth': {
        'controlnet': 'lllyasviel/control_v11f1p_sd15_depth',
        'processor': DepthProcessor,
        'processor_params': {},
        'is_custom': True,
    },
    'softedge': {
        'controlnet': 'lllyasviel/control_v11p_sd15_softedge',
        'processor': HEDdetector,  # PidiNetDetector
        'processor_params': {},
        'is_custom': False,
    },
    'pose': {
        'controlnet': 'lllyasviel/sd-controlnet-openpose',
        'processor': OpenposeDetector,
        'processor_params': {},
        'is_custom': False,
    },
    'norm': {
        'controlnet': 'lllyasviel/control_v11p_sd15_normalbae',
        'processor': NormalBaeDetector,
        'processor_params': {},
        'is_custom': False,
    },
}
