"""
Code adapted from previous method Found: https://github.com/valeoai/FOUND
"""

import argparse
import os
import time
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets.datasets import build_dataset
from tqdm import tqdm
from misc import bbox_iou, get_bbox_from_segmentation_labels
import json

class UODDataset(Dataset):
    def __init__(self, pred_dir, gt_dir):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir

        self.filenames =sorted(os.listdir(pred_dir))

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        
        pred_path = os.path.join(self.pred_dir, self.filenames[idx])
        gt_path = os.path.join(self.gt_dir, self.filenames[idx])

        pred = Image.open(pred_path).convert('L')
        gt = Image.open(gt_path).convert('L')


        pred = np.array(pred)
        gt = np.array(gt)

        return pred, gt
    
def evaluation_unsupervised_object_discovery(
    dataset,
    predict_folder="",
    evaluation_mode: str = 'single', # choices are ["single", "multi"]
    output_dir:str = "outputs",
    no_hards:bool = False,
):
    """
    Evaluate the performance of unsupervised object discovery algorithm.

    Args:
        dataset: The dataset object containing the images and ground truth annotations.
        predict_folder: The folder path where the predicted masks are stored.
        evaluation_mode: The evaluation mode. Choices are ["single", "multi"].
        output_dir: The output directory to save the evaluation results.
        no_hards: Whether to discard images with no ground truth annotations.

    Returns:
        None
    """
    
    assert evaluation_mode == "single"

    sigmoid = nn.Sigmoid()

    # Rest of the code...
def evaluation_unsupervised_object_discovery(
    dataset,
    predict_folder="",
    evaluation_mode: str = 'single', # choices are ["single", "multi"]
    output_dir:str = "outputs",
    no_hards:bool = False,
):
    
    
    assert evaluation_mode == "single"

    sigmoid = nn.Sigmoid()

    # ----------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    start_time = time.time()
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # # Move to gpu
        img = img.cuda(non_blocking=True)
        
        # ------------ GROUND-TRUTH -------------------------------------------
        gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

        if gt_bbxs is not None:
            # Discard images with no gt annotations
            # Happens only in the case of VOC07 and VOC12
            if gt_bbxs.shape[0] == 0 and no_hards:
                continue

        mask_name= im_name.split(".")[0]+".png"

        mask_path_predict = os.path.join(predict_folder, mask_name)

        mask_predict = Image.open(mask_path_predict).convert("L")

        mask_predict = (np.array(mask_predict) > 0).astype(np.uint8)

        mask_predict_tensor = torch.from_numpy(mask_predict)


        # get bbox
        pred = get_bbox_from_segmentation_labels(
            segmenter_predictions=mask_predict_tensor,
            scales=[8,8],
            initial_image_size=init_image_size[1:],
        )

        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred


        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
       
        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")
    
    # 将字典中的所有 NumPy 数组转换为列表
    for key in preds_dict:
        if isinstance(preds_dict[key], np.ndarray):
            preds_dict[key] = preds_dict[key].tolist()

    # 现在可以安全地将字典转换为 JSON 并保存
    with open('preds.json', 'w') as file:
        json.dump(preds_dict, file)
    # Evaluate
    print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
    result_file = os.path.join(output_dir, 'results.txt')
    with open(result_file, 'w') as f:
        f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
    print('File saved at %s'%result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Unsupervised Object Discovery Performance")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--predict_folder", type=str, required=True, help="Folder containing the predicted masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation results")
    
    args = parser.parse_args()

    # Build the validation set
    val_dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        dataset_set=None,
        for_eval=True,
        evaluation_type="uod",
    )
    print(f"\nBuilding dataset {val_dataset.name} (#{len(val_dataset)} images)")

    # Evaluation
    evaluation_unsupervised_object_discovery(
        dataset=val_dataset,
        predict_folder=args.predict_folder,
        output_dir=args.output_dir,
        no_hards=False,  # Assuming this is a default argument; modify as needed
    )
