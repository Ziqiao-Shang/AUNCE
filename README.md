AUNCE
=
- Paper title : Learning Contrastive Feature Representations for Facial Action Unit Detection

Requirements
=
- Python 3
- torch >= 1.4.0
- torchvision
- pillow
- numpy
- tqdm
- timm
- easydict
- pyyaml == 5.4.1

- Check the required python packages in `requirements.txt`.
```
pip install -r requirements.txt
```
Data and Data Prepareing Tools
=
The Datasets we used:
  * [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
  * [DISFA](http://mohammadmahoor.com/disfa-contact-form/)

We provide tools for prepareing data in ```tool/```.
After Downloading raw data files, you can use these tools to process them, aligning with our protocals.


**Training with ImageNet pre-trained models**

Make sure that you download the ImageNet pre-trained models to `checkpoints/` (or you alter the checkpoint path setting in `model/swin_transformer.py`)

The download links of pre-trained models are in `checkpoints/checkpoints.txt`

Training and Testing
=
- to train our approach on BP4D Dataset, run:
```
python train_graph_au.py --dataset "BP4D" --exp_name "Graphau_bp4d_swin_nce_step_1" --fold 1 --gpu_ids '0' --info_nce 'enhance' 
```

- to train our approach on DISFA Dataset, run:
```
python train_graph_au.py --dataset "DISFA" --exp_name "Graphau_disfa_swin_nce_step_1" --fold 1 --gpu_ids '0' --info_nce 'enhance' 
```

- to perform linear evaluation on BP4D Dataset, run:
```
python linear_graph_au.py --dataset "BP4D" --exp_name "Linear_graphau_bp4d_swin_nce_step_1" --fold 1 --gpu_ids '0' --au_num_classes 12
```

- to perform linear evaluation on DISFA Dataset, run:
```
python linear_graph_au.py --dataset "DISFA" --exp_name "Linear_graphau_disfa_swin_nce_step_1" --fold 1 --gpu_ids '0' --au_num_classes 8
```
Main Results
=
**BP4D**

|   Method   |   Source   | AU1 | AU2 | AU4 | AU6 | AU7 | AU10 | AU12 | AU14 | AU15 | AU17 | AU23 | AU24 | Avg. |
| :-------: | :-----------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SimCLR |  2020 ICML  | 38.0 | 36.4 | 37.2 | 66.6 | 64.7 | 76.2 | 76.2 | 51.1 | 29.8 | 56.1 | 27.5 | 37.7 | 49.8 |
| MoCo |  2020 CVPR  | 30.8 | 41.3 | 42.1 | 70.2 | 70.4 | 78.7 | 82.5 | 53.3 | 25.2 | 59.1 | 31.5 | 34.3 | 51.6 |
| EAC-Net |  2018 TPAMI  | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 | 35.9 | 35.8 | 55.9 |
| ROI |  2017 CVPR  | 36.2 | 31.6 | 43.4 | 77.1 | 73.7 | 85.0 | 87.0 | 62.6 | 45.7 | 58.0 | 38.3 | 37.4 | 56.4 |
| ARL |  2019 TAC  | 45.8 | 39.8 | 55.1 | 75.7 | 77.2 | 82.3 | 86.6 | 58.8 | 47.6 | 62.1 | 47.4 | 55.4 | 61.1 |
| MCM |  2024 WACV  | 44.3 | 40.8 | 51.3 | 77.8 | 70.9 | 82.9 | 87.3 | 65.5 | 50.8 | 63.1 | 48.1 | 54.5 | 61.4 |
| MAL |  2023 TAC  | 47.9 | 49.5 | 52.1 | 77.6 | 77.8 | 82.8 | 88.3 | 66.4 | 49.7 | 59.7 | 45.2 | 48.5 | 62.2 |
| CLP |  2023 TIP  | 47.7 | 50.9 | 49.5 | 75.8 | 78.7 | 80.2 | 84.1 | 67.1 | 52.0 | 62.7 | 45.7 | 54.8 | 62.4 |
| JÂA-Net |  2021 IJCV  | 53.8 | 47.8 | 58.2 | 78.5 | 75.8 | 82.7 | 88.2 | 63.7 | 43.3 | 61.8 | 45.6 | 49.9 | 62.4 |
| MMA-Net |  2023 PRL  | 52.5 | 50.9 | 58.3 | 76.3 | 75.7 | 83.8 | 87.9 | 63.8 | 48.7 | 61.7 | 46.5 | 54.4 | 63.4 |
| HMP-PS |  2021 CVPR  | 53.1 | 46.1 | 56.0 | 76.5 | 76.9 | 82.1 | 86.4 | 64.8 | 51.5 | 63.0 | 49.9 | 54.5 | 63.4 |
| GeoConv |  2022 PR  | 48.4 | 44.2 | 59.9 | 78.4 | 75.6 | 83.6 | 86.7 | 65.0 | 53.0 | 64.7 | 49.5 | 54.1 | 63.6 |
| AAR |  2023 TIP  | 53.2 | 47.7 | 56.7 | 75.9 | 79.1 | 82.9 | 88.6 | 60.5 | 51.5 | 61.9 | 51.0 | 56.8 | 63.8 |
| SEV-Net |  2021 CVPR  | 58.2 | 50.4 | 58.3 | 81.9 | 73.9 | 87.7 | 87.5 | 61.6 | 52.6 | 62.2 | 44.6 | 47.6 | 63.9 |
| BTFM |  2023 CVPR  | 57.4 | 52.6 | 64.6 | 79.3 | 81.5 | 82.7 | 85.6 | 67.9 | 47.3 | 58.0 | 47.0 | 44.9 | 64.1 |
| SupHCL |  2022 ACM MM  | 52.8 | 45.7 | 61.6 | 79.5 | 79.3 | 84.7 | 86.9 | 67.6 | 51.4 | 62.5 | 48.6 | 52.3 | 64.4 |
| KSRL |  2022 CVPR  | 53.3 | 47.4 | 56.2 | 79.4 | 80.7 | 85.1 | 89.0 | 67.4 | 55.9 | 61.9 | 48.5 | 49.0 | 64.5 |
| ISTR |  2021 ACM MM  | 54.0 | 46.0 | 55.7 | 79.4 | 78.8 | 84.5 | 87.0 | 67.0 | 55.6 | 63.1 | 50.7 | 55.3 | 64.8 |
| MEGraph |  2022 IJCAI  | 52.7 | 44.3 | 60.9 | 79.9 | 80.1 | 85.3 | 89.2 | 69.4 | 55.4 | 64.4 | 49.8 | 55.1 | 65.5 |
| GLEE-Net |  2024 TCSVT  | 60.6 | 44.4 | 61.0 | 80.6 | 78.7 | 85.4 | 88.1 | 64.9 | 53.7 | 65.1 | 47.7 | 58.5 | 65.7 |
| CLEF |  2023 ICCV  | 55.8 | 46.8 | 63.3 | 79.5 | 77.6 | 83.6 | 87.8 | 67.3 | 55.2 | 63.5 | 53.0 | 57.8 | 65.9 |
| AUNCE(Ours) |       -      | 53.6 | 49.8 | 61.6 | 78.4 | 78.8 | 84.7 | 89.6 | 67.4 | 55.1 | 65.4 | 50.9 | 58.0 | 66.1 |

**DISFA**

|   Method   |   Source   | AU1 | AU2 | AU4 | AU6 | AU9 | AU12 | AU25 | AU26 | Avg. |
| :-------: | :-----------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SIMCLR |  2020 ICML  | 21.2 | 23.3 | 47.5 | 42.4 | 35.5 | 66.8 | 81.5 | 52.7 | 46.4 |
| MoCo |  2020 CVPR  | 22.7 | 18.2 | 45.9 | 45.4 | 34.1 | 72.9 | 83.4 | 54.5 | 47.1 |
| EAC-Net |  2018 TPAMI  | 41.5 | 26.4 | 66.4 | 50.7 | 80.5 | 89.3 | 88.9 | 15.6 | 48.5 |
| ROI |  2017 CVPR  | 41.5 | 26.4 | 66.4 | 50.7 | 80.5 | 89.3 | 88.9 | 15.6 | 48.5 |
| ARL |  2019 TAC  | 43.9 | 42.1 | 63.6 | 41.8 | 40.0 | 76.2 | 95.2 | 66.8 | 58.7 |
| CLP |  2023 TIP  | 42.4 | 38.7 | 63.5 | 59.7 | 38.9 | 73.0 | 85.0 | 58.1 | 57.4 |
| MAL |  2023 TAC  | 43.8 | 39.3 | 68.9 | 47.4 | 48.6 | 72.7 | 90.6 | 52.6 | 58.0 |
| BTFM |  2023 CVPR  | 41.5 | 44.9 | 60.3 | 51.5 | 50.3 | 70.4 | 91.3 | 55.3 | 58.2 |
| SEV-Net |  2021 CVPR  | 55.3 | 53.1 | 61.5 | 53.6 | 38.2 | 71.6 | 95.7 | 41.5 | 58.8 |
| HMP-PS |  2021 CVPR  | 38.0 | 45.9 | 65.2 | 50.9 | 50.8 | 76.0 | 93.3 | 67.6 | 61.0 |
| ISTR |  2021 ACM MM  | 47.5 | 53.3 | 64.4 | 51.8 | 44.4 | 74.7 | 92.1 | 60.7 | 61.1 |
| GeoConv |  2022 PR  | 65.5 | 65.8 | 67.2 | 48.6 | 51.4 | 72.6 | 80.9 | 44.9 | 62.1 |
| MEGraph |  2022 IJCAI  | 52.5 | 45.7 | 76.1 | 51.8 | 46.5 | 76.1 | 92.9 | 57.6 | 62.4 |
| JÂA-Net |  2021 IJCV  | 62.4 | 60.7 | 67.1 | 41.1 | 45.1 | 73.5 | 90.9 | 67.4 | 63.5 |
| SupHCL |  2022 ACM MM  | 52.5 | 58.8 | 70.0 | 53.5 | 51.4 | 73.1 | 95.6 | 58.0 | 64.1 |
| AAR |  2023 TIP  | 62.4 | 53.6 | 71.5 | 39.0 | 48.8 | 76.1 | 91.3 | 70.6 | 64.2 |
| MCM |  2024 WACV  | 49.6 | 44.1 | 67.2 | 65.5 | 49.0 | 81.5 | 85.9 | 71.8 | 64.3 |
| KSRL |  2022 CVPR  | 60.4 | 59.2 | 67.5 | 52.7 | 51.5 | 76.1 | 91.3 | 57.7 | 64.5 |
| CLEF |  2023 ICCV  | 64.3 | 61.8 | 68.4 | 49.0 | 55.2 | 72.9 | 89.9 | 57.0 | 64.8 |
| GLEE-Net |  2024 TCSVT  | 61.9 | 54.0 | 75.8 | 45.9 | 55.7 | 77.6 | 92.9 | 60.0 | 65.5 |
| MMA-Net |  2023 PRL  | 63.8 | 54.8 | 73.6 | 39.2 | 61.5 | 73.1 | 92.3 | 70.5 | 66.0 |
| AUNCE(Ours) |       -      | 61.8 | 58.9 | 74.9 | 49.7 | 56.2 | 73.5 | 92.1 | 64.2 | 66.4 |

Pretrained models
=
The trained models can be downloaded [here](https://drive.google.com/drive/folders/19gQasYDALVbtEr-J9SLFqexjrFuRUtYi?usp=sharing).

Citation
=
Our paper will come soon.

Demos
=
Our single-image demos will come soon.

