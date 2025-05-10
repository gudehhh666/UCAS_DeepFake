# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import torch
import numpy as np
import os
from pathlib import Path
from engine_finetune import test_binary_video_frames
from util.datasets import build_dataset, build_transform
import models_vit
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import time

class CustomDataset:
    def __init__(self, data_path, label_file, is_train, args):

        self.data = []
        self.transform = build_transform(is_train, args)
        print(self.transform)
        self.labels = set()  # To store unique labels


        if label_file:
            self.is_label_file = True
        # If the label_file provides the relative path, join it with dataset_abs_path
            if data_path is not None:
                with open(label_file, 'r') as file:  # .txt file
                    for line in file:
                        # Split the line based on the provided delimiter
                        path, label = line.strip().split(args.delimiter_in_spilt)
                        if path.startswith('/') or path.startswith('\\'):
                            path = path.lstrip('/\\')
                        data_path = os.path.join(data_path, path)
                        self.data.append((data_path, int(label)))
                        self.labels.add(int(label))  # Add label to the set

            # If the label_file provides the absolute path, use it directly
            else:
                with open(label_file, 'r') as file:  # .txt file
                    for line in file:
                        # Split the line based on the provided delimiter
                        path, label = line.strip().split(args.delimiter_in_spilt)
                        self.data.append((path, int(label)))
                        self.labels.add(int(label))  # Add label to the set
        else:
            self.is_label_file = False
            for file in os.listdir(data_path):
                self.data.append((os.path.join(data_path, file), 0))
                # self.labels.add(0)

    def __len__(self):
        return len(self.data)

    def nb_classes(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_label_file:
            img_path, label = self.data[idx]
             # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            return image, label
        else:
            img_path, _ = self.data[idx]
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path.split('/')[-1]


def test_dataset(data_loader, model, device):
    model.eval()

    
    output_all = []
    img_names_all = []
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            images, img_names = batch
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                output = model(images).to(device, non_blocking=True)
                output = output.tolist()
            
            img_names = list(img_names) if not isinstance(img_names, list) else img_names
            img_names_all.extend(img_names)
            output_all.extend(output)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")
    return output_all, img_names_all, end_time - start_time

def get_args_parser():
    parser = argparse.ArgumentParser('FSFM-3C simple evaluation', add_help=False)
    parser.add_argument('--model', default='vit_base_patch16', type=str, help='Model name')
    parser.add_argument('--resume', default='/home/custom/FSFM-main/fsfm-3c/pretrain/checkpoint/pretrained_models/FF++_o_c23_ViT-B/checkpoint-te-400.pth', type=str, help='Path to model weights')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for default: empty for default: ./checkpoint/{user}/experiments_test/from_{FT_folder_name}/{PID}/')
    parser.add_argument('--log_dir', default='',
                        help='path where to save, empty for default: empty for default: ./checkpoint/{user}/experiments_test/from_{FT_folder_name}/{PID}/')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--output_file', default='/home/custom/data/test1/test.txt', type=str, help='File to save results')
    parser.add_argument('--nb_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--device', default='cuda', type=str, help='Device for inference')
    parser.add_argument('--delimiter_in_spilt', default=' ', type=str, help='Delimiter for label file split')
    parser.add_argument('--normalize_from_IMN', action='store_true',
                        help='cal mean and std from imagenet, else from pretrain datasets')
    parser.add_argument('--apply_simple_augment', action='store_true',
                        help='apply simple augment')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--data_path', default='/home/custom/data/test1/processed', type=str,
                        help='data path')
    parser.add_argument('--result_path', default='/home/custom/data/test1', type=str, help='File to save results')
    return parser

def main(args):
    device = torch.device(args.device)
    # Load model
    model = models_vit.__dict__[args.model](num_classes=args.nb_classes)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    dataset_test = CustomDataset(data_path=args.data_path, label_file=None, is_train=False, args=args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    print(f"Testing dataset {args.data_path}, total {len(dataset_test)} images...")
    
    output, img_names, inference_time = test_dataset(data_loader_test, model, device)


    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)

    # 构建results字典
    results = {
        "predictions": dict(zip(img_names, output)),
        "time": inference_time
    }

    # 保存为Excel
    writer = pd.ExcelWriter(os.path.join(result_path, "Donot_Push_Us.xlsx"))
    prediction_frame = pd.DataFrame(
        data={
            "img_names": list(results["predictions"].keys()),
            "predictions": list(results["predictions"].values()),
        }
    )
    time_frame = pd.DataFrame(
        data={
            "Data Volume": [len(results["predictions"])],
            "Time": [results["time"]],
        }
    )
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    print(f"Prediction results for {args.data_path} have been saved to {os.path.join(result_path, 'predictions.xlsx')}")
    del dataset_test, data_loader_test

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
