# GAN-study


This Repository contains things I have created in order to study Generative Adversarial Networks(GAN)[[1]](#1).  
이 레포지토리에는 제가 GAN을 공부하기 위해서 만들었던 것들을 모아두었습니다.

Currently, there are two projects related to GAN, and both projects used *Facial Expression Recognition* (FER) datasets as inputs.

## MoG and GAN

In this project, I wanted to find out the relationship between the input dataset and the performance of GAN.
With Two FER datasets, I tried to analyse the properties of these datasets using different techniques. One of these techniques is to creaste histogram graphs of the dataset to see number of **Mixture of Gaussians** (MoG) that the dataset contain.
Details can be seen in the project's repository.

## CatGAN_mod

In this project, I tried to modify the CatGAN [[2]](#2).
Details can be seen in the project's repository.

## References
<a id="1">[1]</a>  I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio. Generative Adversarial Nets. *Adv Neural Inf Process Syst*, 2014, 27.  
[Github link](https://github.com/goodfeli/adversarial)

<a id="2">[2]</a>  J. T. Springenberg, “Unsupervised and Semi-Supervised Learning with Categorical Generative Adversarial Networks,” *ICLR* 2016.
