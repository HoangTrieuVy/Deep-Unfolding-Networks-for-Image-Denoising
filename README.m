# Deep unfolding neural networks for image denoising

> **Hoang Trieu Vy Le, Nelly Pustelnik, Marion Foare,**
*The faster proximal algorithm, the better unfolded deep learning architecture ? The study case of image denoising,*
EUSIPCO 2021, [Download]([http://bigwww.epfl.ch/publications/hohm1501.pdf](https://hal.archives-ouvertes.fr/hal-03621538/document))

## Data settings:
1. Download BSDS500 [Download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)
2. Extract and place the downloaded "../BSDS500/data" in folder "BSDS500".
3. Generate noisy dataset (auto noise level sigma=50, need to add folder to generate other noisy dataset): 
     python create_training_testing_data.py
     
## Install requirements:
pip install -r requirements.txt

## Training and testing
- model: DnCNN, unfolded_ISTA, unfolded_FISTA, unfolded_CP_v2, unfolded_CP_v3

Example: python main.py --model unfolded_CP_v2 --F 21 --K 13 --lr 1e-4 --batch_size 10 --sigma 50 --num_epochs 500

## Plot results:
Using matlab for visualizing results:
	- Choose model name and parameters corresponding to previous step and run
 					plot_results_unrolling_model.m
