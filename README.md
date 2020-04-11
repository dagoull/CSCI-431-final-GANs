# CSCI-431-final-GANs

The paper associated with this project can be found in this repository: [https://github.com/jrtechs/computer-vision-GANs-paper](https://github.com/jrtechs/computer-vision-GANs-paper).
This repository houses the code used to run the experiments described in this paper. This project was submitted as a RIT CSCI-431 project for professor Sorkunlu's class. 

## Abstract:

Generative Adversarial Networks have emerged as a powerful and customizable class of machine learning algorithms within the past half a decade. They learn the distribution of a dataset for the purposes of generating realistic synthetic samples. It is an active field of research with massive improvements yearly, addressing fundamental limitations of the class and improving on the quality of generated figures. GANs have been successfully applied to music synthesis, face generation, and text-to-image translation.

Within this work, we will look at a variety of GAN architectures and how they compare qualitatively on the popular MNIST dataset. We will explore how differing architectures affect time of convergence, quality of the resulting images, and complexity in training. The theoretical justifications and shortcomings of each methodology will be explored in detail, such that an intuition can be formed on choosing the right architecture for a problem.


# Installation

The dependencies can be installed using pip and our requirements file.
Note: you must be in this director in order for it to work.

```bash
sudo pip3 install -r requirements.txt
```

# Running

When you run on script, it will save the image generated every 400 batches. The loss functions are logged using [TensorBoard](https://www.tensorflow.org/tensorboard/).
This repo implements three different GAN algorithms using [Pytorch](https://pytorch.org). Training and running will take a considerable amount of time.

## GAN

```bash
cd gan
python3 gan.py
```

## DCGAN

```bash
cd dcgan
python3 dcgan.py
```

## WGAN

```bash
cd wgan
python3 wgan.py
```
