import cv2
import numpy as np
from PIL import Image, ExifTags

def detect_gan_artifacts(image_path: str) -> float:
    """Frequency domain analysis (FFT)."""
    img = cv2.imread(image_path, 0)
    if img is None:
        return 0.0
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    # High frequency noise often indicates GANs. Return ratio of high freq energy.
    h, w = img.shape
    center_h, center_w = h // 2, w // 2
    high_freq_mask = np.ones((h, w))
    high_freq_mask[center_h-20:center_h+20, center_w-20:center_w+20] = 0
    high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
    total_energy = np.sum(magnitude_spectrum)
    if total_energy == 0: return 0.0
    return high_freq_energy / total_energy

def check_exif_metadata(image_path: str) -> dict:
    """AI images lack real camera EXIF."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return {"has_exif": False, "ai_probability": 0.9}
        return {"has_exif": True, "ai_probability": 0.1}
    except Exception:
        return {"has_exif": False, "ai_probability": 0.5}
