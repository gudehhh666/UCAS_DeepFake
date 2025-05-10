# Face Swapping with FaceFusion

This project performs automated face swapping using [FaceFusion](https://github.com/facefusion/facefusion) with the `inswapper_128` model and `gfpgan_1.4` for facial enhancement.

## 📁 Input Structure

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

## ⚙️ Script Overview

The script [`process_faces.py`](process_faces.py) automates:
- Iterating through all subfolders under `output_frames/`
- Running FaceFusion in headless mode
- Using `inswapper_128` for face swapping
- Using `gfpgan_1.4` for facial enhancement (blend = 100%)
- Saving output images to the `output/` directory

## ▶️ Run Instructions

Ensure FaceFusion is installed and configured. Then, execute:

```bash
python process_faces.py
```

## 📦 Output

For each input folder, a single processed image is saved to:
```
output/{folder_name}.png
```

## 🧠 Models Used

- **Face Swapper**: `inswapper_128`
- **Face Enhancer**: `gfpgan_1.4`
- **Enhancement Blend Ratio**: 100%
