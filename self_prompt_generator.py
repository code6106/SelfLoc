import argparse
import os
import cv2
import torch
from tqdm import tqdm
from src.object_discovery import ncut
from src.datasets import ImageDataset
import dino.vision_transformer as vits
import numpy as np
from src.visualizations import visualize_img, visualize_predictions
from typing import Any, Dict, List
from datasets.datasets import build_dataset

from segment_anything import SamPredictor, sam_model_registry

def get_vit_encoder(vit_arch, vit_model, vit_patch_size, enc_type_feats):
    if vit_arch == "vit_small" and vit_patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        initial_dim = 384
    elif vit_arch == "vit_small" and vit_patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        initial_dim = 384
    elif vit_arch == "vit_base" and vit_patch_size == 16:
        if vit_model == "clip":
            url = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
        elif vit_model == "dino":
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        initial_dim = 768
    elif vit_arch == "vit_base" and vit_patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        initial_dim = 768
    # print(vit_model,"vit_model")
    if vit_model == "dino":
        vit_encoder = vits.__dict__[vit_arch](patch_size=vit_patch_size, num_classes=0)
        # TODO change if want to have last layer not unfrozen
        for p in vit_encoder.parameters():
            p.requires_grad = False
        vit_encoder.eval().cuda()  # mode eval
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )
        vit_encoder.load_state_dict(state_dict, strict=True)

        hook_features = {}
        if enc_type_feats in ["k", "q", "v", "qkv", "mlp"]:
            # Define the hook
            def hook_fn_forward_qkv(module, input, output):
                hook_features["qkv"] = output

            vit_encoder._modules["blocks"][-1]._modules["attn"]._modules[
                "qkv"
            ].register_forward_hook(hook_fn_forward_qkv)
    else:
        raise ValueError("Not implemented.")

    return vit_encoder, initial_dim, hook_features



@torch.no_grad()
def extract_feats(dims, type_feats="k"):
# Extract the qkv features of the last attention layer
    nb_im, nh, nb_tokens, _ = dims
    qkv = (
        hook_features["qkv"]
        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1) 
    # Modality selection
    if type_feats == "k":
        #feats = k[:, 1:, :]
        feats = k
    elif type_feats == "q":
        #feats = q[:, 1:, :]
        feats = q
    elif type_feats == "v":
        #feats = v[:, 1:, :]
        feats = v
    return feats

def draw_boxes_on_image(image, boxes, color=(255, 0, 0), thickness=2):
    """
    Draw boxes on an image.

    Parameters:
    image (numpy array): The image on which boxes will be drawn.
    boxes (numpy array): An array of boxes, each box is represented by 4 points (x1, y1, x2, y2).
    color (tuple): The color of the box lines. Default is red.
    thickness (int): The thickness of the box lines.

    Returns:
    numpy array: The image with boxes drawn on it.
    """
    # Iterate over all boxes and draw each one on the image
    for box in boxes:
        # Extract coordinates
        x1, y1, x2, y2 = box
        # Draw rectangle on the image
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image

def save_mask_as_image(masks, image_name, save_path, color=None):
    """
    Save a NumPy mask as an image to a specified directory with a specific name.

    Parameters:
    mask (numpy.ndarray): The mask to be saved as an image.
    image_name (str): The name of the original image (e.g., 'a.jpg').
    save_path (str): The directory where the image will be saved.
    color (tuple): The color to apply to the mask. Default is None.

    Returns:
    None
    """
    save_full_path = os.path.join(save_path, image_name)

    os.makedirs(save_full_path, exist_ok=True)
    
    for i, mask in enumerate(masks):
        # Ensure the mask is a binary mask
        mask = (mask > 0).astype(np.uint8)

        # If a color is provided, apply it to the mask
        if color is not None:
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                colored_mask[:, :, i] = mask * color[i]
        else:
            colored_mask = mask * 255

        # Construct the file name by adding the suffix "_sam" before the file extension
        save_file_name = f"{i}.png"

        # Save the mask as an image in the specified directory
        # print(colored_mask.shape)
        cv2.imwrite(os.path.join(save_full_path, save_file_name), colored_mask)


def generate(image_path,dataset_dir="/data/datasets",save_path="./output/"):

    # ------------------------------------
    
    #load datasets
    if image_path is not None:
        dataset = ImageDataset(image_path, None)
    else:
        dataset = build_dataset(
            root_dir=dataset_dir,
            dataset_name="DUTS-TEST",
            dataset_set=None,
            for_eval=True,
            evaluation_type="saliency",
        )

        dataset.fullimg_mode()
        batch_size = 1
        valloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    pbar = dataset.dataloader
    for im_id, inp in enumerate(pbar):
        img = inp[0]
        # print(img.shape,type(img),"img.shape", "type(img)")
        init_image_size = img.shape
        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # print(im_name,"im_name")
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / vit_patch_size) * vit_patch_size),
            int(np.ceil(img.shape[2] / vit_patch_size) * vit_patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded
        
        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // vit_patch_size
        h_featmap = img.shape[-1] // vit_patch_size

        # Encoder forward pass
        att = vit_encoder.get_last_selfattention(img[None, :, :, :])
        # print(att.shape,"att.shape")
        feats = extract_feats(dims=att.shape, type_feats=enc_type_feats)
        # print(feats.shape," feats.shape")

        # Scaling factor
        scales = [vit_patch_size, vit_patch_size]
        
        # print(feats.shape,"feats.shape")
        #apply NCUT
        pred, objects, foreground, seed , bins, eigenvector= ncut(
            feats, [w_featmap, h_featmap], 
            scales, 
            init_image_size, 
            0.2, 
            1e-5, 
            im_name=im_name, 
            no_binary_graph=False)
        

        
        image = dataset.load_image(im_name, size_im)
        
        # visualize_predictions(image, pred, "./output/vis_prompt", im_name)
        
        image=np.array(image)
        
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=pred)    
        # print(im_name)
        
        save_mask_as_image(masks, im_name,save_path )
        return pred,im_name

def list_files(directory):
    """
    List all files in a given directory.

    Parameters:
    directory (str): The directory to list files from.

    Returns:
    list: A list of full file paths.
    """
    files_list = []
    # Walk through all files and directories within the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            files_list.append(full_path)

    return files_list

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate and save mask predictions.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated mask predictions.")
    parser.add_argument("--images_path", type=str, required=True, help="Directory path containing images to process.")
    
    args = parser.parse_args()

    # DINO configuration
    vit_model="dino"
    vit_arch="vit_small"
    vit_patch_size=16
    enc_type_feats="k"
    vit_encoder, initial_dim, hook_features = get_vit_encoder(
                vit_arch, vit_model, vit_patch_size, enc_type_feats
        )
    
    # SAM configuration
    sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
    device = torch.device('cuda')
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Load and process images
    all_files = list_files(args.images_path)
    preds_dict = {}

    for file in tqdm(all_files, desc="Generating all prompt sam pred result"):
        pred, im_name = generate(image_path=file, dataset_dir=args.images_path, save_path=args.save_path)
        

