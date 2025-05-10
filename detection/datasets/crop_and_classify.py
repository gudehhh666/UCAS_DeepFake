import os
import cv2
import pandas as pd
from tqdm import tqdm
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/val')
    parser.add_argument('--gt_file', type=str, default=None)
    args = parser.parse_args()

    IMG_LIST_PATH = os.path.join(args.data_root, 'img_list.txt')
    FACE_INFO_PATH = os.path.join(args.data_root, 'face_info.txt')
    IMG_DIR = os.path.join(args.data_root, 'imgs')
    OUTPUT_ROOT = os.path.join(args.data_root, 'processed')


    with open(IMG_LIST_PATH, 'r') as f:
        img_names = [line.strip() for line in f.readlines()]


    with open(FACE_INFO_PATH, 'r') as f:
        face_infos = [list(map(int, line.strip().split())) for line in f.readlines()]

    if args.gt_file is not None:
        labels = pd.read_excel(args.gt_file, header=None)[1].tolist()
        labels = labels[1:]
        print(len(img_names), len(face_infos), len(labels))
        assert len(img_names) == len(face_infos) == len(labels)

        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        for idx in tqdm(range(len(img_names)), desc='Processing'):
            img_name = img_names[idx]
            x1, y1, x2, y2 = face_infos[idx]
            label = str(labels[idx])
            
            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path):
                print(f'no img: {img_path}')
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f'no img: {img_path}')
                continue
            h, w = img.shape[:2]


            x1_, y1_, x2_, y2_ = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = img[y1_:y2_, x1_:x2_]
            
            label_dir = os.path.join(OUTPUT_ROOT, label)
            os.makedirs(label_dir, exist_ok=True)
            save_path = os.path.join(label_dir, img_name)
            if img is None or img.size == 0:
                print("IMG IS EMPTY!")
            else:
                cv2.imwrite(save_path, img)
        
    else:
        print(f'no gts files given')
        print(len(img_names), len(face_infos))
        assert len(img_names) == len(face_infos) 

        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        for idx in tqdm(range(len(img_names)), desc='Processing'):
            img_name = img_names[idx]
            x1, y1, x2, y2 = face_infos[idx]
            
            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path):
                print(f'no img: {img_path}')
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f'no img: {img_path}')
                continue
            h, w = img.shape[:2]


            x1_, y1_, x2_, y2_ = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = img[y1_:y2_, x1_:x2_]
            
            os.makedirs(OUTPUT_ROOT, exist_ok=True)
            save_path = os.path.join(OUTPUT_ROOT, img_name)
            if img is None or img.size == 0:
                print("IMG IS EMPTY!")
            else:
                cv2.imwrite(save_path, img)