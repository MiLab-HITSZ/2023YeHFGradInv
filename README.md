# HF-GradInv
Implementation of AAAI 2024 paper "High-Fidelity Gradient Inversion in Distributed Learning"

# Requirements

I have tested on:

- PyTorch 1.13.0
- CUDA 11.0


# The Simplest Implementation

### If you want to test the gradient inversion attacks:

> python main_attack.py

![avatar](/custom_data/test_recon/final_rec8.jpg)

### If you want to test with duplicated labels: 

> python main_attack_duplicate_labels.py

![avatar](/custom_data/test_recon/final_rec.jpg)

## Test images, reconstructions, as well as the auxiliary data we used in our paper are available in folder "custom_data".

## It is easy to test on more settings, just need to adjust the corresponding parameters as well as the training data in "custom_data".


 # REFERENCES
 
 *https://github.com/JonasGeiping/breaching*
 
