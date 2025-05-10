# 🛡️🎭 UCAS_DeepFake

![License](https://img.shields.io/badge/license-MIT-green)
![Issues](https://img.shields.io/github/issues/gudehhh666/UCAS_DeepFake)

> This repository contains our **complete solution** for the *AI Security and Adversarial Competition (Spring ’25)* from TEAM **Donot Push_Us**.  
> The project is split into two independent, but complementary, modules:  
>
> 1. **Deepfake Detection** (🛡️)  
> 2. **Deepfake Generation** (🎭)  
>

---

- ## Table of Contents
  - [📦 Get Start](#-get-start)
  - [🛡️ Detection](#-detection)
  - [🎭 Generation](#-generation)
  - [👥 Members & Responsibilities](#-members--responsibilities)

---

### 📦 Get Start

* Download Celeb-DF-v2 from **[celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)**, we did not use any other training data.

* Environment Settings

  * For detection:

  ```bash
  conda create -n deepfake python=3.9
  conda activate deepfake
  pip install -r requirements.txt
  ```

  * For generation, you can refer to the [FaceFusion Installation Instructions](https://docs.facefusion.io/installation)

* Code Base References

  * [FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](https://github.com/wolo-wolo/FSFM.git)

  * [DFGC_startkit](https://github.com/bomb2peng/DFGC_starterkit.git)

  * [FaceFusion](https://github.com/facefusion/facefusion.git)

    We sincerey appreciate the outstanding contribution of the aforementioned  works.

---

### 🛡️ Detection

1. Download pertained model weights:
   ```python
   import os
   os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
   
   from huggingface_hub import snapshot_download, login, hf_hub_download
   
   
   
   # hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/checkpoint-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
   # hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/checkpoint-te-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
   # hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/pretrain_ds_mean_std.txt", local_dir="./checkpoint/", local_dir_use_symlinks=False)
   
   hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/checkpoint-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
   hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/checkpoint-te-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
   hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/pretrain_ds_mean_std.txt", local_dir="./checkpoint/", local_dir_use_symlinks=False)
   
   ```

   

2. Process traing data
   ```python
   python dataset_preprocess.py --dataset CelebDFv2
   ```

   Then you can get the training dataset
   Remember to modify the Celeb-DF path from `datasets/finetune/preprocess/config/default.py`

   ```python
   _C.CelebDFv2_path = '/home/custom/data/Celeb-DF-v2/'
   ```

3. Fine tuning the pretrained model
   ```bash
   cd /home/custom/detection/cross_dataset_DfD
   bash train.sh
   ```

​	We release our checkpoint at [here](Eincasia/deepfake_detection), you need to download the checpoint and `.txt` both and put them 	in a same dictionary.

4. Evaluation
   ```bash 
   cd /home/custom/detection/cross_dataset_DfD
   bash eval.sh
   ```

   Remember before evaluation, you need to process your data use `detection/datasets/crop_and_classify.py`. Given the data_root and gt_file then it can process the valadation and test data to the standard format.

5. Prediction
   ```bash
   cd /home/custom/detection/cross_dataset_DfD
   bash prediction.sh
   ```

   After prediction, you can get a xlsx with image name and log_prob.
   
6.  **`test1` result saved in [`detection/result/test1/Donot_Push_Us.xlsx`](https://github.com/gudehhh666/UCAS_DeepFake/blob/main/detection/result/test1/Donot_Push_Us.xlsx)**



---

### 🎭 Generation

This project performs automated face swapping using [FaceFusion](https://github.com/facefusion/facefusion) with the `inswapper_128` model and `gfpgan_1.4` for facial enhancement.

📁 Input Structure

Each subfolder in the input directory should contain:

- `source*.png`: the image providing the face (source identity)
- `target*.png`: the image where the face will be swapped in

Example folder:

```
output_frames/
└── clip_001/
    ├── source.png
    └── target.png
```

⚙️ Script Overview

The script [`process_faces.py`](process_faces.py) automates:

- Iterating through all subfolders under `output_frames/`
- Running FaceFusion in headless mode
- Using `inswapper_128` for face swapping
- Using `gfpgan_1.4` for facial enhancement (blend = 100%)
- Saving output images to the `output/` directory

▶️ Run Instructions

Ensure FaceFusion is installed and configured. Then, execute:

```bash
python process_faces.py
```

📦 Output

For each input folder, a single processed image is saved to:

```
output/{folder_name}.png
```

🧠 Models Used

- **Face Swapper**: `inswapper_128`
- **Face Enhancer**: `gfpgan_1.4`
- **Enhancement Blend Ratio**: 100%



---
Face Swapping with FaceFusion

This project performs automated face swapping using [FaceFusion](https://github.com/facefusion/facefusion) with the `inswapper_128` model and `gfpgan_1.4` for facial enhancement.

📁 Input Structure

Each subfolder in the input directory should contain:
- `source*.png`: the image providing the face (source identity)
- `target*.png`: the image where the face will be swapped in

Example folder:
```
output_frames/
└── clip_001/
    ├── source.png
    └── target.png
```

⚙️ Script Overview

The script [`process_faces.py`](process_faces.py) automates:
- Iterating through all subfolders under `output_frames/`
- Running FaceFusion in headless mode
- Using `inswapper_128` for face swapping
- Using `gfpgan_1.4` for facial enhancement (blend = 100%)
- Saving output images to the `output/` directory

▶️ Run Instructions

Ensure FaceFusion is installed and configured. Then, execute:

```bash
python process_faces.py
```

📦 Output

For each input folder, a single processed image is saved to:
```
output/{folder_name}.png
```
**Our Submission Output Can download from**

```txt
通过网盘分享的文件：Donot_Push_US.tar.gz
链接: https://pan.baidu.com/s/1Zl9lC1BdIo7O-13TfQEHSA 提取码: y319 
--来自百度网盘超级会员v5的分享
```



🧠 Models Used

- **Face Swapper**: `inswapper_128`
- **Face Enhancer**: `gfpgan_1.4`
- **Enhancement Blend Ratio**: 100%


### 👥Members & Responsibilities
| Memembers  | Responsibilities      |
| ----- | ------------- |
| 王新茗 | Detection     |
| 朱子林 | Detection |
| 范善威 | Detection |
| 龚昱嘉 | Detection |
| 陈家良 | Generation    |
| 李仲正 | Generation |
| 陈鸿 | Generation |
| 曹方成 | Generation |

For any issues  you can contact wangxinming2024@ia.ac.cn.