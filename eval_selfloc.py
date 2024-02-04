import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

# Calculate the Intersection over Union (IoU) of two binary masks
def calculate_miou(pred, gt):
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Calculate the accuracy between prediction and ground truth
def calculate_acc(pred, gt):
    correct = np.sum(pred == gt)
    total = pred.size
    acc = correct / total
    return acc

def main(dataset_name, mask_root, pred_root):
    method = 'SelfLoc'
    iou_list = []
    acc_list = []
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    
    mask_name_list = sorted(os.listdir(mask_root))
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # Add calculation for miou and acc
        miou = calculate_miou(pred, mask)
        acc = calculate_acc(pred, mask)

        iou_list.append(miou)
        acc_list.append(acc)
        
        # Update metrics
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    # Calculate averages and get metric results
    avg_miou = np.mean(iou_list)
    avg_acc = np.mean(acc_list)
    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "miou": avg_miou,
        "acc": avg_acc,
        "maxFm": fm["curve"].max(),
        "meanFm": fm["curve"].mean(),
        "wFmeasure": wfm,
        "Smeasure": sm,
        "maxEm": em["curve"].max(),
        "meanEm": em["curve"].mean(),
        "MAE": mae,
    }

    print(results)
    with open("evalresults.txt", "a") as file:
        file.write(method + ' ' + dataset_name + ' ' + str(results) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation metrics.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--mask_root", type=str, required=True, help="Path to the ground truth masks.")
    parser.add_argument("--pred_root", type=str, required=True, help="Path to the predicted masks.")
    
    args = parser.parse_args()
    main(args.dataset_name, args.mask_root, args.pred_root)
