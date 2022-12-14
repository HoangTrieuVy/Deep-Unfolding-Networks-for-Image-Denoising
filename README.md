# Deep unfolding neural networks for image denoising

> **Hoang Trieu Vy Le, [Nelly Pustelnik](https://perso.ens-lyon.fr/nelly.pustelnik/), [Marion Foare](https://perso.ens-lyon.fr/marion.foare/),**
*The faster proximal algorithm, the better unfolded deep learning architecture ? The study case of image denoising,*
EUSIPCO  Belgrade, Serbie, 29 Aug - 2 Sept. 2022, [Download](https://hal.archives-ouvertes.fr/hal-03621538/document)

## <div align="center">Quick start Examples </div>

<details open>
<summary>Install</summary>

Clone the repository and install [requirements.txt](https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/blob/main/requirements.txt) in a
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
Pretrained Pytorch code to perform image denoising considering either **DnCNN** or the unfolded schemes proposed in our EUSIPCO paper: **unfolded_ISTA**, **unfolded_FISTA**,  **unfolded_CP**(without SC), **unfolded_ScCP** (strong convexity)}.


<summary>Remarks</summary>

* Pretrained models can be found in [Checkpoints](https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/tree/main/checkpoints).
* The pretrained DnCNN contains F=9 and K=13
* The pretrained unfolded_ISTA contains F=21 and K= 13
* The pretrained unfolded_FISTA contains F=21 and K= 13
* The pretrained unfolded_CP contains F=21 and K= 13
* The pretrained unfolded_ScCP contains F=21 and K= 13
* All the pretrained architectures listed above are comparable in terms of number of neurons.
* In our experiments, the most efficient scheme is unfolded_ScCP so we advise the user to denoise with this unfolded scheme.

  
**Inference Settings**
 ```python
optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  DnCNN,unfolded_ISTA, unfolded_FISTA, unfolded_CP, unfolded_ScCP
  --i I          original impage path
  --n N          noisy impage path
 ```


```bash
cd examples

python unrolled_NN_denoiser.py --model [choice_of_model]  --n [your_noisy_image]

python unrolled_NN_denoiser.py --model unfolded_ScCP --i 10081.jpg --n 10081_noisy.jpg
```

<img align="center" width="1500" src="https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/blob/main/examples/results_unfolded_ScCP_10081_noisy.jpg" >


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


```python
python main.py --model unfolded_CP_v2 --F 21 --K 13 --lr 1e-4 --batch_size 10 --sigma 50 --num_epochs 500
```

**Plot results**

Using matlab for visualizing results:
- Choose model name and parameters corresponding to previous step and run

```matlab
plot_results_unrolling_model.m
```
<img align="center" width="1800" src="https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/blob/main/print_results/compare.jpg" >

<img align="center" width="500" src="https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/blob/main/print_results/PSNR_test.jpg" >

<img align="center" width="500" src="https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/blob/main/print_results/loss_train.jpg" >



</details>


