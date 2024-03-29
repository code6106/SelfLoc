from src.merge_base import Merger
import os
from PIL import Image
import numpy as np
from pathlib import Path

class Merge_MaxIoU(Merger):
    def __init__(self, params, num_cls=21, threshold=0.2):
        super(Merge_MaxIoU, self).__init__(params, num_cls)
        self.threshold = threshold

    def merge(self, predict, name, sam_folder, save_path):
        seen = []
        processed_mask = np.zeros_like(predict)
        candidate_names = []  
        for i in range(1, self.num_cls):
            pre_cls = predict == i
            if np.sum(pre_cls) == 0:
                continue
            iou = 0
            candidates = []
            sam_mask = np.zeros_like(pre_cls)
            for filename in os.scandir(sam_folder):
                if filename.is_file() and filename.path.endswith('png') and filename.path not in seen:
                    cur_sam = np.array(Image.open(filename.path)) == 255
                    sam_mask = np.logical_or(sam_mask, cur_sam)
                    improve_thresh = 2 * np.sum((pre_cls == cur_sam) * pre_cls) - np.sum(cur_sam)
                    
                    improve_pred_thresh = np.sum((pre_cls == cur_sam) * pre_cls) / np.sum(pre_cls)
                    # print(f'Improve: {improve_thresh}, {improve_pred_thresh}, {filename.path}')
                    if improve_thresh > 0 or improve_pred_thresh >= 0.85:
                        # print(f'Improve: {improve_thresh}, {improve_pred_thresh}, {filename.path}')
                        candidates.append(cur_sam)
                        candidate_names.append(filename.name)
                        seen.append(filename.path)
                        iou += np.sum(pre_cls == cur_sam)
            cam_mask = np.logical_and(sam_mask==0, pre_cls==1)
            
            # Trust CAM if SAM has no prediction on that pixel
            candidates.append(cam_mask)
            processed_mask[np.sum(candidates, axis=0) > 0] = i
        # print(candidate_names)
        im = Image.fromarray(processed_mask)
        im.save(f'{save_path}/{name}.png')