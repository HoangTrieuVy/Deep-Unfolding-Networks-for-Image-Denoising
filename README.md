# Deep unfolding neural networks for image denoising

> **Hoang Trieu Vy Le, [Nelly Pustelnik](https://perso.ens-lyon.fr/nelly.pustelnik/), [Marion Foare](https://perso.ens-lyon.fr/marion.foare/),**
*The faster proximal algorithm, the better unfolded deep learning architecture ? The study case of image denoising,*
EUSIPCO 2022, [Download](https://hal.archives-ouvertes.fr/hal-03621538/document)

## <div align="center">Quick start Examples </div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/blob/main/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising  # clone
cd your_repo
pip install -r requirements.txt  # install
```

</details>
<details open>
<summary>Inference Denoiser</summary>

Choose a pretrained model among {DnCNN, unfolded_ISTA, unfolded_FISTA, unfolded_CP_v2 (strong convexity), unfolded_CP_v3(without SC)} in [Checkpoints](https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/tree/main/checkpoints)

```python
python deep_unfolding_denoiser_sig50.py --model DnCNN --F 13 --K 9 --sigma 50 --batch_size 10 --num_epochs 500

python deep_unfolding_denoiser_sig50.py --model unfolded_ISTA --F 21 --K 13 --sigma 50 --batch_size 10 --num_epochs 500

python deep_unfolding_denoiser_sig50.py --model unfolded_FISTA --F 21 --K 13 --sigma 50 --batch_size 10 --num_epochs 500

python deep_unfolding_denoiser_sig50.py --model unfolded_CP_v2 --F 21 --K 13 --sigma 50 --batch_size 10 --num_epochs 500

python deep_unfolding_denoiser_sig50.py --model unfolded_CP_v3 --F 21 --K 13 --sigma 50 --batch_size 10 --num_epochs 500

```

</details>

<details open>
<summary>Training</summary>

**Data settings**

1. [Download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) BSDS500 
2. Extract and place the downloaded "../BSDS500/data" in folder "BSDS500".
3. Generate noisy dataset (auto noise level sigma=50, need to add folder to generate other noisy dataset): 

```python
python create_training_testing_data.py
```

**Training and testing** ( some specific parameters )
- model: DnCNN, unfolded_ISTA, unfolded_FISTA, unfolded_CP_v2, unfolded_CP_v3
- F: size of linear operator or number of features of convolution operators
- K: depth of networks
- sigma: noise level (default=50)

```python
python main.py --model unfolded_CP_v2 --F 21 --K 13 --lr 1e-4 --batch_size 10 --sigma 50 --num_epochs 500
```

**Plot results**

Using matlab for visualizing results:
- Choose model name and parameters corresponding to previous step and run

```matlab
plot_results_unrolling_model.m
```
</details>


