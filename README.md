
# Multimodal Hyperbolic Multiple Instance Learning for Video Anomaly Detection
This is an official repo for 'Multimodal Graph Hyperbolic MIL for Video Anomaly Detection'  
The implementation is based on [TEVAD: Improved video anomaly detection with captions](https://github.com/coranholmes/TEVAD)
and [Hyperbolic Image Embeddings](https://github.com/leymir/hyperbolic-image-embeddings)



<img width="4477" height="1196" alt="GraphHypVAD" src="https://github.com/user-attachments/assets/34a7a8e2-91b2-4cef-8794-4ae82a28c495" />


## Requirements

This repo is based on the PyTorch framework, with Python version 3.9.  
To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets 
Create folders **data** and **list** inside the main project folder.  

Follow [this](https://github.com/coranholmes/TEVAD/blob/main/README.md#visual-features) to generate or download visual features.   
Follow [this](https://github.com/coranholmes/TEVAD/blob/main/README.md#text-features) to generate or download text features.  

Save visual and text features in the folder **data**
```bash
├── Crime
│   ├── sent_emb_n
│   └── UCF_ten_crop_i3d_v1
├── ShanghaiTech
│   ├── feats
│   ├── feats_SH
│   └── sent_emb_n
└── UCSD_Ped2
    ├── ped2_ten_crop_i3d
    └── sent_emb_n


```
Get ground truths from [here](https://onedrive.live.com/?authkey=%21AHLyTWgDJJOW24A&id=2159F1430FCCC356%21538526&cid=2159F1430FCCC356&sb=name&sd=1)  
Save lists and ground truths in the folder **list**:
```bash
├── gt-ped2.npy
├── gt-sh2.npy
├── gt-ucf.npy
├── ped2-i3d.list
├── ped2-i3d-test.list
├── shanghai-i3d-test-10crop.list
├── shanghai-i3d-train-10crop.list
├── ucf-i3d.list
└── ucf-i3d-test.list
```
Alter paths in the lists according to the path of the **data** directory.


## Training and Evaluation

To train and evaluate the model, run this command (e.g on the ShanghaiTech benchmark):

```train and evaluate
python main.py --config ./configs/sh_config.yaml 
```
To train on a different benchmark, replace the config from the **configs** folder accordingly.
