from typing import Any
from PointCloudRenderer.pointcloud_renderer import PointCloudRenderer
from PointCloudRenderer.rgbd_dataloader import get_dataloader, get_dataset_and_cache
from PointCloudRenderer.point_to_rgb_transformer import TransformerPointsToRGBModule
import random, cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

class Averaging_Renderer():
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # Extract the input tensor
        points = args[0]

        # Calculate the average RGB values along the second axis (num_points)
        average_rgb = torch.mean(points[:, :, 3:], dim=1)

        # Return the result with shape (batch_size, 3)
        return average_rgb

def main():
    torch.set_default_dtype(torch.float)
    data_dir = get_dataset_and_cache()
    dataset = get_dataloader(data_dir).dataset

    index = random.randint(0, len(dataset)-1)

    image_path = dataset.image_paths[index]
    depth_path = dataset.depth_paths[index]
    extrinsics_path = dataset.extrinsics_paths[index]
    intrinsics_path = dataset.intrinsics_paths[index]
    debug = False
    

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    

    extrinsics = np.loadtxt(extrinsics_path)
    if extrinsics.ndim == 1:
        extrinsics = extrinsics.reshape((3, 4))
    elif extrinsics.shape != (3, 4):
        raise ValueError("Extrinsics matrix should be a 3x4 matrix or a 1D array with 12 values.")
    extrinsics = torch.from_numpy(extrinsics).to("cuda")

    intrinsics = np.loadtxt(intrinsics_path)
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.reshape((3, 3))
    elif intrinsics.shape != (3, 3):
        raise ValueError("Intrinsics matrix should be a 3x3 matrix or a 1D array with 9 values.")
    intrinsics = torch.from_numpy(intrinsics).to("cuda")

    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # Convert the raw depth values to meters using the https://github.com/ankurhanda/sunrgbd-meta-data
    depth_img = depth_img / 10000

    depth_pixels = [(y, x) for y in range(depth_img.shape[0]-1) for x in range(depth_img.shape[1]-1)]
    # Then translate those pixels into a pointcloud in 3d space using the camera extrinsic and intrinsic matrix
    pointcloud = dataset.get_pointcloud(depth_pixels, depth_img, img, extrinsics, intrinsics)
    pointcloud = torch.tensor(pointcloud)
    
    averager = Averaging_Renderer()
    renderer = PointCloudRenderer(pointcloud, averager, intrinsics, extrinsics)

    images = renderer.render_images(img_resolution_x=300, img_resolution_y=300)
    plt.imshow(img)
    plt.show()

    plt.imshow(images[0].cpu().detach().numpy().astype(int))
    plt.show()


if __name__ == "__main__":
    main()

