import numpy as np
import random


def uniform_d10(depth_img, image_pixel, k_points): # Sample k_points pixels in the depth image around the image_pixel
        y, x = image_pixel
        height, width = depth_img.shape
        depth_pixels = []
        for i in range(k_points):
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            sx = max(0, min(width - 1, x + dx))
            sy = max(0, min(height - 1, y + dy))
            depth_pixels.append((sy, sx))
        return depth_pixels
