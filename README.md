# GAN-tensorflow
 
Simple tensorflow implementation of Generative Adversarial Nets which generate MNIST data from TF. 

The paper of GAN is [here](url)

### Model

![image](https://user-images.githubusercontent.com/42796324/47069839-e2c7bb80-d22a-11e8-94c7-f3f0ca5dbfed.png)
 
                                        GAN의 학습 과정 원리 <출처: 네이버랩스> 

## Prerequisition

- Python 3.6
- [Tensorflow 1.0.1](https://github.com/tensorflow/tensorflow/tree/r0.12)

## Run
    $ python3 GAN_train.py

## Result
All results are randomly sampled.

*Name* | *Epoch 1* | *Epoch 50* | *Epoch 100* | *Epoch 200*
:---: | :---: | :---: | :---: | :---: |
GAN | <img src = 'generated_image/000.png' height = '230px'> | <img src = 'generated_image/010.png' height = '230px'> | <img src = 'generated_image/020.png' height = '230px'> | <img src = 'generated_image/039.png' height = '230px'>



