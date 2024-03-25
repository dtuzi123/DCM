# DCM

>📋 This is the implementation of Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Memory

>📋 Accepted by CVPR 2024

# Title : Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Memory

# Paper link : 


# Abstract

Online Task-Free Continual Learning (OTFCL) aims to learn novel concepts from streaming data without accessing task information. Memory-based approaches have shown remarkable results in OTFCL, but most require accessing supervised signals to implement their sample selection mechanisms, limiting their applicability for unsupervised learning. In this study, we address this issue by proposing a novel memory management approach, namely the Dynamic Cluster Memory (DCM), which builds new memory clusters to capture distribution shifts over time without accessing any supervised signal. 
DCM introduces a novel memory expansion mechanism based on the knowledge discrepancy criterion, which evaluates the novelty of the incoming data as the signal for the memory expansion, ensuring a compact memory capacity. We also propose a new sample selection approach that automatically stores incoming data samples with similar semantic information in the same memory cluster, facilitating the knowledge diversity among memory clusters. Furthermore, a novel memory pruning approach is proposed to automatically remove overlapping memory clusters through a graph relation evaluation, ensuring a fixed memory capacity while maintaining the diversity among the samples stored in the memory. The proposed DCM is model-free, plug-and-play, and can be used in both supervised and unsupervised learning without modifications. Empirical results on OTFCL experiments show that the proposed DCM outperforms the state-of-the-art while memorizing fewer data samples.

# Environment

1. Pytorch 1.12
2. Python 3.7

Our code is based on the improved diffusion model ("https://github.com/openai/improved-diffusion")

# Training and evaluation

>📋 Python xxx.py, the model will be automatically trained and then report the results after the training.

>📋 Different parameter settings of OCM would lead different results and we also provide different settings used in our experiments.

# BibTex
>📋 If you use our code, please cite our paper as:
