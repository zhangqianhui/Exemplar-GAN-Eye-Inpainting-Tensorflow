# Exemplar_GAN_Eye_Inpainting
The tensorflow reference implement of [Eye In-Painting with Exemplar Generative Adversarial Networks](https://arxiv.org/abs/1712.03999)  

# Introduction

[Paper Introduction](https://github.com/bdol/exemplar_gans#introduction).

## Network Architecture

<p align="center">
  <img src="/images/net.jpg">
</p>

## Paper result

<p align="center">
  <img src="/images/paper_result.jpg">
</p>

# Differences with the original paper.

- Because the origin paper does't provide the details of model, this implement use network architecture and hyperparamters from [SG-GAN](https://github.com/zhangqianhui/Sparsely_Grouped_GAN)(Using adapted residual image learning and use spectral norm)

- Just use refernece image as the exemplar, not code.

## Dependencies
* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [Tensorflow 1.4+](https://github.com/tensorflow/tensorflow)


## Usage

- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/SG_GAN_tensorflow.git
cd SG_GAN_tensorflow
```
- Download the CeleA-ID dataset

You can download CelebA-ID Benchmark dataset from [Google drive](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0) or download according to the [Dataset](https://github.com/bdol/exemplar_gans#celeb-id-benchmark-dataset) 

and unzip CeleA-ID into a directory. 

- Train the model using the default parameter
```bash
python main.py --OPER_FLAG=0
```
- Test the model 

```bash
python main.py --OPER_FLAG=1
```

# Our results

## (1)Input image; (2) Reference image; (3) image to in-paint; (4) in-painted image; (5) local real eye region; (6) generated eye region;

<p align="center">
  <img src="/images/our_result.jpg">
</p>

## Reference code

[Sparsely_Grouped_GAN](https://github.com/zhangqianhui/Sparsely_Grouped_GAN)

[DCGAN tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

[Spectral Norm tensorflow](https://github.com/taki0112/Spectral_Normalization-Tensorflow)



