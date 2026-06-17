import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def percentile_stretch(image, low=2, high=98):
    """
    Stretches the intensity of an image based on percentiles to remove outliers.
    """
    if image.ndim == 3:
        stretched = np.zeros_like(image)
        for i in range(image.shape[-1]):
            stretched[..., i] = percentile_stretch(image[..., i], low, high)
        return stretched.astype(np.uint8)
    
    low_val = np.percentile(image, low)
    high_val = np.percentile(image, high)
    
    stretched = np.clip(image, low_val, high_val)
    stretched = (stretched - low_val) * 255.0 / (high_val - low_val + 1e-5)
    
    return stretched.astype(np.uint8)
