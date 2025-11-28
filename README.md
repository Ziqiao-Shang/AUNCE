# AUNCE
- Paper title : Learning Contrastive Feature Representations for Facial Action Unit Detection

Introduction
=
For the Facial Action Unit (AU) detection task, accurately capturing the subtle facial differences between distinct AUs is essential for reliable detection. Additionally, AU detection faces challenges from class imbalance and the presence of noisy or false labels, which undermine detection accuracy. In this paper, we introduce a novel contrastive learning framework aimed for AU detection that incorporates both self-supervised and supervised signals, thereby enhancing the learning of discriminative features for accurate AU detection. To tackle the class imbalance issue, we employ a negative sample re-weighting strategy that adjusts the step size of updating parameters for minority and majority class samples. Moreover, to address the challenges posed by noisy and false AU labels, we employ a sampling technique that encompasses three distinct types of positive sample pairs. This enables us to inject self-supervised signals into the supervised signal, effectively mitigating the adverse effects of noisy labels. Our experimental assessments, conducted on five widely-utilized benchmark datasets (BP4D, DISFA, BP4D+, GFT and Aff-Wild2), underscore the superior performance of our approach compared to state-of-the-art methods of AU detection. 

<img width="1683" height="561" alt="38da8ee2be2d67ecdf0b1ad2a487f609" src="https://github.com/user-attachments/assets/fb4994df-aa53-4a1f-8544-0b32dbc1f0bc" />



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

Training
=
- to train our approach on BP4D Dataset, run:
```
python train_graph_au.py --dataset "BP4D" --exp_name "Graphau_bp4d_swin_nce_step_1" --fold 1 --gpu_ids '0' --info_nce 'enhance' 
```

- to train our approach on DISFA Dataset, run:
```
python train_graph_au.py --dataset "DISFA" --exp_name "Graphau_disfa_swin_nce_step_1" --fold 1 --gpu_ids '0' --info_nce 'enhance' 
```

Testing
=
We adhere to the established linear evaluation protocol, as commonly employed in previous studies [CPC](https://arxiv.org/abs/1807.03748) and [Simclr](https://proceedings.mlr.press/v119/chen20j.html). Please refer to this [link](https://github.com/google-research/simclr).

BP4D_Sequence_split 
= 
Fold1: 'F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014','F023','M008'

Fold2: 'M011','F003','M010','M002','F005','F022','M018','M017','F013','M016','F020','F011','M013','M005'

Fold3: 'F007','F015','F006','F019','M006','M009','F012','M003','F004','F021','F017','M015','F014'

DISFA_Sequence_split 
= 
Fold1: 'SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016'

Fold2: 'SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024'

Fold3: 'SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004'

Pretrained models
=
The trained models can be downloaded [here](https://drive.google.com/drive/folders/19gQasYDALVbtEr-J9SLFqexjrFuRUtYi?usp=sharing).

Citation
=
If you use this code for your research, please cite our paper
```
@article{shang2024learning,
  title={Learning contrastive feature representations for facial action unit detection},
  author={Shang, Ziqiao and Liu, Bin and Lv, Fengmao and Teng, Fei and Li, Tianrui and Guo, Lanzhe},
  journal={arXiv preprint arXiv:2402.06165},
  year={2024}
}
```
