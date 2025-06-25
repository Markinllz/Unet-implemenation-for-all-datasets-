![wdwed]https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fmachine-learning%2Fu-net-architecture-explained%2F&psig=AOvVaw3jyV2XHDgkvBhAQqqOwZ2S&ust=1750897206127000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCODZ4MGmi44DFQAAAAAdAAAAABAE![image](https://github.com/user-attachments/assets/efdef042-5527-44b3-901e-a0dab9ec74ee)


# U-Net: Retina Blood Vessel Segmentation with PyTorch

Customized implementation of U-Net in PyTorch for semantic segmentation of retinal blood vessels from fundus images.

---

## Quick Start

### Requirements

- Python 3.6 or newer
- PyTorch 1.13 or newer
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Data

- Download and prepare your dataset of retinal fundus images and corresponding vessel masks.
- Place images in `data/Data/train/image/` and masks in `data/Data/train/mask/`.
- For testing, use `data/Data/test/image/` and `data/Data/test/mask/`.

**Note:**  
- Input images should be RGB.
- Masks should be binary (vessels = 1, background = 0), saved as PNG files.

---

## Training

To train the U-Net model on your RetinaDataset, run:

```bash
python train.py --epochs 20 --batch-size 4 --learning-rate 1e-4 --scale 1.0
```

**Optional arguments:**
- `--epochs` Number of training epochs (default: 5)
- `--batch-size` Batch size (default: 1)
- `--learning-rate` Learning rate (default: 1e-5)
- `--scale` Downscaling factor for images (default: 0.5)
- `--validation` Percent of data used for validation (default: 10)
- `--amp` Use mixed precision (recommended for modern GPUs)

---

## Prediction

After training and saving your model to a `.pth` file, you can predict vessel masks for new images:

To predict a single image and save the result:
```bash
python predict.py --model data/Data/checkpoints/checkpoint_epoch5.pth -i data/Data/test/image/1.png -o data/Data/test/mask/1_pred.png
```

To predict multiple images and visualize the results without saving:
```bash
python predict.py --model data/Data/checkpoints/checkpoint_epoch5.pth -i data/Data/test/image/1.png data/Data/test/image/2.png --viz --no-save
```

**Optional arguments:**
- `--model` Path to the trained model checkpoint
- `--input` Input image(s) for prediction
- `--output` Output mask(s) to save
- `--viz` Visualize predictions
- `--no-save` Do not save output masks
- `--mask-threshold` Threshold for binarizing the predicted mask (default: 0.5)
- `--scale` Downscaling factor for input images (should match training)

---

## Evaluation

You can evaluate your model using metrics such as Dice coefficient, Intersection over Union (IoU), and pixel-wise accuracy.  
Ground truth masks and predicted masks should be compared pixel-wise.

---

## Weights & Biases

Training progress, loss curves, validation metrics, and predicted masks can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  
Simply run your training script and follow the link printed in the console.

---

## Dataset

The Retina Blood Vessel Segmentation dataset contains high-resolution retinal fundus images with pixel-level vessel annotations.  
Masks are binary: vessel pixels are labeled as 1, background as 0.

---

## Citation

If you use this code or dataset, please cite the original U-Net paper:

> Olaf Ronneberger, Philipp Fischer, Thomas Brox:  
> U-Net: Convolutional Networks for Biomedical Image Segmentation

---
