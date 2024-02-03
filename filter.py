import os
import cv2
import numpy as np

def load_masks(mask_folder):
    """
    Load all masks from a given folder.

    Parameters:
    mask_folder (str): The folder containing mask files.

    Returns:
    list of numpy.ndarray: List of loaded masks.
    """
    masks = []
    for mask_file in os.listdir(mask_folder):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_folder, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append((mask, mask_file))
    return masks

def filter_masks(masks, foreground_mask, threshold=0.1):
    """
    Filter out masks and categorize them as background or foreground.

    Parameters:
    masks (list of numpy.ndarray): List of masks from the folder.
    foreground_mask (numpy.ndarray): The rough mask of the foreground object.
    threshold (float): Threshold for determining background.

    Returns:
    tuple: Two lists of tuples, one for background masks and one for foreground masks, each with their filenames.
    """
    background_masks = []
    foreground_masks = []
    for mask, filename in masks:
        # Calculate mask_folder  with the foreground mask
        intersection = np.logical_and(mask > 0, foreground_mask > 0)
        intersection_rate = np.sum(intersection) / np.sum(mask > 0)

        # Categorize the mask
        if intersection_rate < threshold:
            background_masks.append((mask, filename))
        else:
            foreground_masks.append((mask, filename))
    
    return background_masks, foreground_masks

# 使用示例
mask_folder = '/data/user/2022/zjh/datasets/DUTS/DUTS-TE-Sam/ILSVRC2012_test_00000003'  # 替换为 mask 文件夹路径
masks = load_masks(mask_folder)

foreground_mask = cv2.imread('/data/user/2022/zjh/Seuol/output/DUTS-TE/ILSVRC2012_test_00000003/0.png', cv2.IMREAD_GRAYSCALE)  # 加载前景 mask
background_masks, foreground_masks = filter_masks(masks, foreground_mask)

# 输出背景和前景 mask 的文件名
print("Background masks:")
for mask, filename in background_masks:
    print(f"  {filename}")

print("\nForeground masks:")
for mask, filename in foreground_masks:
    print(f"  {filename}")

