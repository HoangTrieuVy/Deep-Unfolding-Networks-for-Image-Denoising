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
   
Choose a pretrained model among {*DnCNN*, *unfolded_ISTA*, *unfolded_FISTA*, *unfolded_CP_v2* (strong convexity), *unfolded_CP_v3*(without SC)} in [Checkpoints](https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/tree/main/checkpoints) and unfolding networks contains various parameters which are:
  
**Inference Settings**
 ```python
optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  DnCNN,unfolded_ISTA, unfolded_FISTA, unfolded_CP, unfolded_ScCP
  --i I          original impage path
  --n N          noisy impage path
 ```


```python
cd examples
   
python example.py --model ScCP --i 10081_noisy.jpg
```

<img align="center" width="1500" src="https://github.com/HoangTrieuVy/Deep-Unfolding-Networks-for-Image-Denoising/blob/main/examples/10081_result.jpg" >


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


