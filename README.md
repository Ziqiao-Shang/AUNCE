AUNCE
=
- Paper title : Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition
- Arxiv : Our paper has been submitted to Arxiv, the link will come soon.

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

Thanks to the offical Pytorch and [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

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

Pretrained models
=
The trained models can be downloaded [here](https://drive.google.com/drive/folders/1UNyjXUiALkJO42WKh9hiVAQD_BN4XoD2?usp=sharing).

Citation
=
The paper will come soon.
