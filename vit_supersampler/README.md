# Vision Transformer based SuperSampler

## 1. Dataset
- Use CelebFaces Attributes Dataset(CelebA).
- **202,599 face images** of the size 178×218.
- Train with 90% and test with 10% of whole dataset.

## 2. Objective
- Understand architecture of **Vision Transformer**.
- Understand how **Multi-head Self Attention** works and visulize it.
- Classify face images by gender.

## 3. Train
Results after one epoch with 32 batch size which equals to 5,698 iters. MSE loss for upsampling with bicubic method is about 0.0075, as goal of this experiment.

### Trial 01
- 5.72it/s, total loss=3.52 for 634 iters
- Without positional Embedding and skip connect upsampled input image by nearest method.
- Loss after about 6,500 iters is 0.00539, which is smaller than bicubic upsampling error.

### Trial 02
- 5.79it/s, total loss=3.33
- Without positional Embedding and skip connect upsampled input image by bicubic method.

### Trial 03
- 5.25it/s, total loss=3.27
- With positional Embedding and skip connect upsampled input image by bicubic method.
- After 3 epochs, total loss decrease to 2.89 and stop training.