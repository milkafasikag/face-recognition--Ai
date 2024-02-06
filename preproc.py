from skimage import exposure
import cv2
import numpy as np


def resize_images(images, new_size=(250, 250)):
    resized_images = [cv2.resize(image, new_size) for image in images]
    return resized_images


def normalize_images(images):
    normalized_images = [image / 255.0 for image in images]
    return normalized_images


def apply_histogram_equalization(images):
    equalized_images = [exposure.equalize_hist(image) for image in images]
    return equalized_images


def apply_gaussian_blur(images, sigma=1.0):
    blurred_images = [cv2.GaussianBlur(image, (5, 5), sigma) for image in images]
    return blurred_images


def flatten(images):
    flattened_images = [image.flatten() for image in images]
    return np.array(flattened_images)


def preprocess_images(images):
    images = resize_images(images)
    images = normalize_images(images)
    images = apply_histogram_equalization(images)
    images = apply_gaussian_blur(images)
    images = flatten(images)
    return images
