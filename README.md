# Overhead-MNIST Image Generation Using Generative Adversarial Networks (GAN)

A deep learning project that generates synthetic grayscale images of ships and helicopters from the Overhead-MNIST dataset using Conditional Generative Adversarial Networks (cGAN). The project applies a baseline MLP-based GAN and a modified DCGAN architecture, with quantitative evaluation using the Fréchet Inception Distance (FID).

## Highlights

- Image preprocessing: grayscale conversion, resizing to 28x28, and pixel normalization to [-1, 1]
- Stratified train/validation/test split (80/10/10) preserving class balance
- Conditional GAN design using label embeddings so the generator can be conditioned on class (ship vs. helicopter)
- Baseline GAN using fully connected (MLP) generator and discriminator
- Modified DCGAN architecture using ConvTranspose2D/Conv2D layers, Batch Normalization, increased latent dimension, larger label embedding, and tailored weight initialization
- Hyperparameter tuning including Adam optimizer, BCE loss, and one-sided label smoothing
- Quantitative image quality evaluation using Fréchet Inception Distance (FID) via InceptionV3 features
- Visual comparison of generated samples between baseline and modified models

## Data

The dataset consists of grayscale images from two classes drawn from the Overhead-MNIST dataset:
- Ship: 8,012 images
- Helicopter: 5,906 images

Total: 13,918 images, resized to 28x28 pixels and normalized to a [-1, 1] pixel range.   
Data was split into training (80%), validation (10%), and test (10%) sets using stratified sampling, then wrapped in PyTorch `Dataset/DataLoader` objects for GAN training.

## Model Experiments
Two conditional GAN architectures were developed and compared, both conditioned on class labels via embedding layers:
- Baseline GAN: A fully connected (MLP) generator (Linear layers with LeakyReLU, Tanh output) and discriminator (Linear layers with LeakyReLU, Sigmoid output), trained for 200 epochs with Adam (lr=2e-4) and standard BCE loss.
- Modified GAN (DCGAN): A convolutional architecture incorporating:
  - A DCGAN-style generator combining a linear projection with ConvTranspose2D and Conv2D layers to better capture spatial structure
  - A convolutional discriminator using Conv2D layers instead of fully connected layers, for more effective spatial feature extraction
  - Batch Normalization after linear/transposed-convolution layers (generator) and convolution layers (discriminator) for training stability
  - Increased latent dimension (100 → 128) and label embedding dimension (50 → 64) for richer representation capacity
  - Tailored weight initialization: Normal initialization for convolutional layers, Xavier initialization for linear layers
  - Hyperparameter tuning: Adam optimizer (lr=2e-4), BCE loss, and one-sided label smoothing (real label = 0.9) to prevent an overly confident discriminator

## Results

The baseline GAN failed to converge to a stable equilibrium: discriminator loss steadily decreased while generator loss increased over training, and generated images remained largely unrecognizable noise patterns, consistent with its high FID score of 291.16.

The modified DCGAN showed much more stable adversarial training, with discriminator loss decreasing gradually while generator loss stayed in a competitive range. Generated images showed clearly recognizable ship and helicopter silhouettes with reasonable diversity, though some samples still appeared blurry or lacked fine detail. This is reflected in the FID score dropping to 64.59, a 77.82% improvement over the baseline, confirming that the DCGAN's convolutional structure, batch normalization, richer label embeddings, and tuned hyperparameters produced a synthetic image distribution much closer to the real data distribution.

Modified GAN (DCGAN) was selected as the final model based on its substantially better FID score and visibly more realistic generated samples.

## How to Run

The notebook was developed using Python and Jupyter Notebook (originally run on Google Colab with GPU).

1. Clone this repository:

```
git clone https://github.com/feliceeeee/Overhead-MNIST_Image_Generation_Using_GAN.git
```

2. Install the required libraries:

```
pip install numpy matplotlib pillow scikit-learn torch torchvision tensorflow scipy jupyter
```

3. Ensure the dataset (Overhead-MNIST ship and helicopter images) is extracted to: `data/version2/train` dan `data/version2/test`, each containing `ship/` and `helicopter/` subfolders
4. Open the notebook: `notebook/Overhead-MNIST Image Generation Using Generative Adversarial Networks (GAN).ipynb`
5. Run all cells to perform data preprocessing, baseline and modified GAN training, image generation, and FID-based evaluation.
