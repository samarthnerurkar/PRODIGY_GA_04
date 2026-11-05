# 🏢 Conditional GAN (pix2pix style) on CMP Facades

A lightweight PyTorch implementation of a conditional GAN for image-to-image translation. It learns to turn colored architectural label maps into realistic building facade images.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-red)
![Colab](https://img.shields.io/badge/Google-Colab-orange)

## ✨ Features

- Automatic dataset download (CMP Facades from the Berkeley pix2pix datasets)
- Whole dataset loaded into GPU memory once for fast training
- U-Net generator with skip connections
- PatchGAN discriminator
- Combined adversarial and L1 loss
- Visual comparison of Input (Label), Ground Truth and Generated images

## 🛠 Tech Stack

- Python, PyTorch, torchvision
- Pillow, Matplotlib
- Google Colab (T4 GPU)

## ⚡ Getting Started

1. Open `cGAN.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run all cells. The dataset downloads and extracts automatically.

## 🧠 Model

**Generator (light U-Net)**
- Encoder of strided convolutions, decoder of transposed convolutions
- Skip connections concatenate encoder features into the decoder
- Final layer uses `Tanh`

**Discriminator (PatchGAN)**
- Takes the input label and the real or generated image together
- Convolution blocks with BatchNorm and LeakyReLU(0.2)
- Outputs a grid of real/fake predictions per image patch

## 🎛 Training Settings

| Setting | Value |
|---------|-------|
| Image size | 128 × 128 |
| Batch size | 16 |
| Epochs | 40 |
| L1 weight (`LAMBDA`) | 100 |
| Optimizer | Adam, lr = 2e-4, betas = (0.5, 0.999) |
| Losses | `BCEWithLogitsLoss` + `L1Loss` |

## 📊 Results

Training log (sample):

```
Epoch 25/40 | D 0.412 | G 24.798
Epoch 30/40 | D 0.319 | G 23.492
Epoch 35/40 | D 0.482 | G 24.671
Epoch 40/40 | D 0.564 | G 19.990
```

The generated facades follow the layout of the input label map (windows, doors, balconies) but are still blurry, since the model is small and trained for only 40 epochs at low resolution.

## 🔄 Workflow

<img width="1737" height="486" alt="image" src="https://github.com/user-attachments/assets/69d781a4-b5b6-433e-afee-3154e0d037a3" />


## 📚 Experiment Library

Settings to try for better results (suggestions, not from the notebook):

| Experiment | Change | Expected effect |
|------------|--------|-----------------|
| Higher resolution | `SIZE = 256` | Sharper details, slower training |
| Longer training | `EPOCHS = 100` | Less blurry output |
| Lower L1 weight | `LAMBDA = 10` | More texture, less faithful layout |
| Bigger batch | `BATCH = 32` | Smoother training, more GPU memory |

## ⚠️ Notes

- Each dataset image contains the real photo and the label map side by side, which the notebook splits automatically.
- For sharper results, increase `SIZE` and `EPOCHS`, or use the full pix2pix architecture.

## 📚 References

- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix)
- CMP Facades dataset

## 📄 License

MIT License
