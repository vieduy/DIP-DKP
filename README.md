# Diffusion-based Kernel Prior for Blind Super-Resolution (DIP-DKP)
## Requirements
- Python 3.6, PyTorch >= 1.6 
- Requirements: opencv-python, tqdm
- Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.5


## Quick Run
To run the code without preparing data, run this command:
```bash
cd DIPDKP
python main.py --SR --sf 4 --dataset Test
```

---

## Data Preparation
To prepare testing data, please organize images as `data/datasets/DIV2K/HR/0801.png`, and run this command:
```bash
cd data
python prepare_dataset.py --model DIPDKP --sf 2 --dataset Set5
```
## DIP-DKP

To test DIP-DKP (no training phase), run this command:

```bash
cd DIPDKP
python main.py --SR --sf 2 --dataset Set5
```

## Results
Please refer to the [report](https://drive.google.com/file/d/14xxliqfKwyodURaeJH7hEKkTl1Vtnczz) for results. DIP-DKP is randomly intialized, different runs may get slightly different results. The reported results are averages of 5 runs.


## License & Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [normalizing_flows](https://github.com/kamenbliznashki/normalizing_flows), [DIP](https://github.com/DmitryUlyanov/deep-image-prior), [KernelGAN](https://github.com/sefibk/KernelGAN) and [USRNet](https://github.com/cszn/KAIR). Please also follow their licenses. Thanks for their great works.
