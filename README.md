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

   After prediction, you can get a txt with prediction label and log_prob like:
   ```txt
   0.png,0.622559,1
   1.png,0.597168,1
   2.png,0.635254,1
   3.png,0.636230,1
   4.png,0.631348,1
   5.png,0.625488,1
   6.png,0.714355,1
   7.png,0.614258,1
   ```

---

### 🎭 Generation
---

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