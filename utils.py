from numpy import *
from scipy.signal import *
import yaml
import numpy as np
from scipy.ndimage import gaussian_filter


def get_config(f):
    with open(f,'r') as s:
        return yaml.load(s)


def attack(clean_images, sigma=1.0):

    adversarial_images = []

    for original_image in clean_images:
        perturbation = gaussian_filter(np.random.randn(*original_image.shape), sigma)
        adversarial_image = original_image + perturbation
        adversarial_image = np.clip(adversarial_image, 0, 255).astype(np.uint8)
        adversarial_images.append(adversarial_image)

    return adversarial_images
