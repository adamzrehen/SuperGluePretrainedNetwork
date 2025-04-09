import numpy as np

def approximate_intrinsics(image_width, image_height):
    # A common heuristic: choose focal length as the maximum dimension.
    focal_length = max(image_width, image_height)
    # Assume the principal point is at the center of the image.
    cx = image_width / 2.0
    cy = image_height / 2.0
    K = np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1]
    ], dtype=np.float32)
    return K


# Example usage:
image_width, image_height = 1415, 1080  # replace with your image dimensions
K = approximate_intrinsics(image_width, image_height)
print("Approximated K:\n", K)

