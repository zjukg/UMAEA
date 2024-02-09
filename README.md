<!--<div align="center">
  <img src="https://github.com/zjukg/UMAEA/blob/main/IMG/UMAEA2.jpg" alt="Logo" width="600">
</div>-->

# 🏝️ [Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment](https://arxiv.org/abs/2307.16210)
![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/UMAEA/blob/main/LICENSE)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![arxiv badge](https://img.shields.io/badge/arxiv-2307.16210-red)](https://arxiv.org/abs/2307.16210)
[![ISWC2023](https://img.shields.io/badge/ISWC-2023-%23bd9f65?labelColor=%23bea066&color=%23ffffff)](https://iswc2023.semanticweb.org/)

>In the face of modality incompleteness, some models succumb to overfitting the modality noise, and exhibit performance oscillations or declines at high modality missing rates. This indicates that the inclusion of additional multi-modal data can sometimes **adversely affect EA**. To address these challenges, we introduces **`UMAEA`**, a robust multi-modal entity alignment approach designed to tackle **uncertainly missing and ambiguous visual modalities**.

<div align="center">
    <img src="https://github.com/zjukg/UMAEA/blob/main/IMG/case.jpg" width="70%" height="auto" />
</div>

## 🔔 News
- **`2024-02` We preprint our Survey [Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey](http://arxiv.org/abs/2402.05391)  [[`Repo`](https://github.com/zjukg/KG-MM-Survey)].**
- **`2024-01` Our paper [Revisit and Outstrip Entity Alignment: A Perspective of Generative Models](https://arxiv.org/abs/2305.14651) is accepted by ICLR 2024 !**
- **`2023-10` We preprint our paper [Universal Multi-modal Entity Alignment via Iteratively Fusing Modality Similarity Paths](https://arxiv.org/abs/2310.05364) !**
- **`2023-07` We release the [[Repo](https://github.com/zjukg/UMAEA)] for our paper: [Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment](https://arxiv.org/abs/2307.16210) ! [[`Slide`](https://github.com/zjukg/UMAEA/blob/main/Slide/Chen-ISWC-2023.pdf)], ISWC 2023** 
- **`2023-04` We release the complete code and [data](https://drive.google.com/file/d/1VIWcc3KDcLcRImeSrF2AyhetBLq_gsnx/view?usp=sharing) for [MEAformer](https://github.com/zjukg/MEAformer) ! [[`Slide`](https://github.com/zjukg/MEAformer/blob/main/Slide/MEAformer-Slide.pdf)] [[`Vedio`](https://youtu.be/5Kjzg0EPavI)], ACM MM 2023**

## 🔬 Dependencies
```bash
pip install -r requirement.txt
```
#### Details
- Python (>= 3.7)
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- numpy (>= 1.19.2)
- [Transformers](http://huggingface.co/transformers/) (>= 4.21.3)
- easydict (>= 1.10)
- unidecode (>= 1.3.6)
- tensorboard (>= 2.11.0)




## 🚀 Train
- **Quick start**: Using  script file (`run.sh`)
```bash
>> cd UMAEA
>> bash run.sh
```
- **Optional**: Using the `bash command`
- **Model Training Recommendation📍**: For more stable and efficient model training, we suggest using the code `without CMMI` (`w/o CMMI`) initially. If you plan to use this model as a baseline, we also recommend using the script `without CMMI` to directly measure the model's performance in an **End2End** scenario.
```bash
# Command Details:
# Bash file / GPU / Dataset / Data Split / R_{sa} / R_{img}
# Begin:
# ---------- R_{img} = 0.4 & iter. & w/o CMMI ----------
>> bash run_umaea_00.sh 0 OEA_D_W_15K_V1 norm 0.2 0.4
>> bash run_umaea_00.sh 0 OEA_D_W_15K_V2 norm 0.2 0.4
>> bash run_umaea_00.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.4
>> bash run_umaea_00.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.4
>> bash run_umaea_00.sh 0 DBP15K fr_en 0.3 0.4
>> bash run_umaea_00.sh 0 DBP15K ja_en 0.3 0.4
>> bash run_umaea_00.sh 0 DBP15K zh_en 0.3 0.4
# ---------- R_{img} = 0.6 & non-iter. & w/o CMMI ----------
>> bash run_umaea_0.sh 0 OEA_D_W_15K_V1 norm 0.2 0.6
>> bash run_umaea_0.sh 0 OEA_D_W_15K_V2 norm 0.2 0.6
>> bash run_umaea_0.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.6
>> bash run_umaea_0.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.6
>> bash run_umaea_0.sh 0 DBP15K fr_en 0.3 0.6
>> bash run_umaea_0.sh 0 DBP15K ja_en 0.3 0.6
>> bash run_umaea_0.sh 0 DBP15K zh_en 0.3 0.6
# --------- R_{img} = 0.1 & non-iter. & w/ CMMI ---------
>> bash run_umaea_012.sh 0 OEA_D_W_15K_V1 norm 0.2 0.1
>> bash run_umaea_012.sh 0 OEA_D_W_15K_V2 norm 0.2 0.1
>> bash run_umaea_012.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.1
>> bash run_umaea_012.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.1
>> bash run_umaea_012.sh 0 DBP15K fr_en 0.3 0.1
>> bash run_umaea_012.sh 0 DBP15K ja_en 0.3 0.1
>> bash run_umaea_012.sh 0 DBP15K zh_en 0.3 0.1
# --------- R_{img} = 0.2 & iter. & w/ CMMI ---------
>> bash run_umaea_012012.sh 0 OEA_D_W_15K_V1 norm 0.2 0.2
>> bash run_umaea_012012.sh 0 OEA_D_W_15K_V2 norm 0.2 0.2
>> bash run_umaea_012012.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.2
>> bash run_umaea_012012.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.2
>> bash run_umaea_012012.sh 0 DBP15K fr_en 0.3 0.2
>> bash run_umaea_012012.sh 0 DBP15K ja_en 0.3 0.2
>> bash run_umaea_012012.sh 0 DBP15K zh_en 0.3 0.2
```

📌 **Tips**: you can open the `run_umaea_X.sh` file for parameter or training target modification.
- `stage_epoch`: The number of epochs in each stage **( 1 / 2-1 / 2-2 )**
    - E.g., "250,0,0"
- `il_stage_epoch`: The number of epochs in each iterative stage **( 1 / 2-1 / 2-2 )**
    - E.g., "0,0,0"

## 🎯 Standard Results

$\bf{H@1}$ Performance with the Settings: **`w/o surface & Non-iterative`**. We modified part of the [MSNEA](https://github.com/liyichen-cly/MSNEA) to involve not using the content of attribute values but only the attribute types themselves (See [issues](https://github.com/zjukg/MEAformer/issues/3) for details):
| Method | $\bf{DBP15K_{ZH-EN}}$ | $\bf{DBP15K_{JA-EN}}$ | $\bf{DBP15K_{FR-EN}}$ |
|:------------------:|:----------------:|:----------------:|:----------------:|
|        [MSNEA](https://github.com/liyichen-cly/MSNEA)          |    .609     |     .541     |      .557     |
|        [EVA](https://github.com/cambridgeltl/eva)          |    .683     |     .669    |      .686     |
|        [MCLEA](https://github.com/lzxlin/mclea)          |    .726     |     .719     |      .719     |
|        [MEAformer](https://github.com/zjukg/MEAformer)         |    **.772**     |     **.764**     |      **.771**     |
|        [UMAEA](https://github.com/zjukg/umaea)         |    **.800**     |     **.801**     |      **.818**     |

## 📚 Dataset (MMEA-UMVM)
>To create our `MMEA-UMVM` (uncertainly missing visual modality) datasets, we perform **random image dropping** on MMEA datasets. Specifically, we randomly discard entity images to achieve varying degrees of visual modality missing, ranging from 0.05 to the maximum $R_{img}$ of the raw datasets with a step of 0.05 or 0.1. Finally, we get a total number of 97 data split as follow:

<div align="center">
    
Dataset | $R_{img}$ 
:---: | :---:  
$DBP15K_{ZH-EN}$ | $0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.7829~(STD)$ 
$DBP15K_{JA-EN}$ | $0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.7032~(STD)$ 
$DBP15K_{FR-EN}$ | $0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6758~(STD)$ 
$OpenEA_{EN-FR}$ | $0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0~(STD)$
$OpenEA_{EN-DE}$ | $0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0~(STD)$
$OpenEA_{D-W-V1}$ | $0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0~(STD)$ 
$OpenEA_{D-W-V2}$ | $0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0~(STD)$ 

</div>

📍 **Download**: 
- **[ Option ]** The raw Multi-OpenEA images are available at [`Baidu Cloud Drive`](https://pan.baidu.com/s/1oikW9BlutAvfJHcfMLDcDQ) with the pass code **`aoo1`**. We only filter the `RANK NO.1` image for each entity.
- Case analysis Jupyter script: [GoogleDrive](https://drive.google.com/file/d/1AUTo7FhzvRYTsLTrDFOW1NVbsTFlGraM/view?usp=sharing) (**180M**) base on the raw images of DBP15K entities (need to be unzip).
- **[ Option ]** The raw images of entities appeared in DBP15k can be downloaded from [`Baidu Cloud Drive`](https://pan.baidu.com/s/1nRpSLJtTUEXDD4cgfSZZQQ) ((**50GB**)) with the pass code **`mmea`**. All images are saved as title-image pairs in dictionaries and can be accessed with the following code :
```python
import pickle
zh_images = pickle.load(open("eva_image_resources/dbp15k/zh_dbp15k_link_img_dict_full.pkl",'rb'))
print(en_images["http://zh.dbpedia.org/resource/香港有線電視"].size)
```
- 🎯 The training data is available at [GoogleDrive](https://drive.google.com/file/d/1TDESVvXh5eq2aW50qGuqqNajy5Mkc6Nw/view?usp=sharing) (6.09G). Unzip it to make those files **satisfy the following file hierarchy**:
```
ROOT
├── data
│   └── mmkg
└── code
    └── UMAEA
```

#### Code Path
<details>
    <summary>👈 🔎 Click</summary>
    
```
UMAEA
├── config.py
├── main.py
├── requirement.txt
├── run.sh
├── run_umaea_00.sh
├── run_umaea_012012.sh
├── run_umaea_012.sh
├── run_umaea_0.sh
├── model
│   ├── __init__.py
│   ├── layers.py
│   ├── Tool_model.py
│   ├── UMAEA_loss.py
│   ├── UMAEA.py
│   └── UMAEA_tools.py
├── src
│   ├── data.py
│   ├── __init__.py
│   └── utils.py
├── torchlight
│   ├── __init__.py
│   ├── logger.py
│   ├── metric.py
│   └── utils.py
└── tree.txt
```

</details>



#### Data Path
<details>
     <summary>👈 🔎 Click</summary>
    
```
mmkg
├── dump
├── DBP15K
│   ├── fr_en
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   ├── ja_en
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   ├── translated_ent_name
│   │   ├── dbp_fr_en.json
│   │   ├── dbp_ja_en.json
│   │   ├── dbp_zh_en.json
│   │   ├── srprs_de_en.json
│   │   └── srprs_fr_en.json
│   └── zh_en
│       ├── ent_ids_1
│       ├── ent_ids_2
│       ├── ill_ent_ids
│       ├── training_attrs_1
│       ├── training_attrs_2
│       ├── triples_1
│       └── triples_2
├── OpenEA
│   ├── OEA_D_W_15K_V1
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── rel_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   ├── OEA_D_W_15K_V2
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── rel_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   ├── OEA_EN_DE_15K_V1
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── rel_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   ├── OEA_EN_FR_15K_V1
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── rel_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   └── pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.05.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.15.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.1.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.2.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.3.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.45.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.4.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.55.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.5.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.6.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.75.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.7.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.8.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.95.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict_0.9.pkl
│       ├── OEA_D_W_15K_V1_id_img_feature_dict.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.05.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.15.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.1.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.2.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.3.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.45.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.4.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.55.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.5.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.6.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.75.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.7.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.8.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.95.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict_0.9.pkl
│       ├── OEA_D_W_15K_V2_id_img_feature_dict.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.05.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.15.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.1.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.2.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.3.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.45.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.4.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.55.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.5.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.6.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.75.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.7.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.8.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.95.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict_0.9.pkl
│       ├── OEA_EN_DE_15K_V1_id_img_feature_dict.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.05.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.15.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.1.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.2.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.3.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.45.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.4.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.55.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.5.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.6.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.75.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.7.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.8.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.95.pkl
│       ├── OEA_EN_FR_15K_V1_id_img_feature_dict_0.9.pkl
│       └── OEA_EN_FR_15K_V1_id_img_feature_dict.pkl
├── pkls
│   ├── fr_en_GA_id_img_feature_dict_0.05.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.15.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.1.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.2.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.3.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.45.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.4.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.55.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.5.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.6.pkl
│   ├── fr_en_GA_id_img_feature_dict_0.7.pkl
│   ├── fr_en_GA_id_img_feature_dict.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.05.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.15.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.1.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.2.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.3.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.45.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.4.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.55.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.5.pkl
│   ├── ja_en_GA_id_img_feature_dict_0.6.pkl
│   ├── ja_en_GA_id_img_feature_dict.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.05.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.15.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.1.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.2.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.3.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.45.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.4.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.55.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.5.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.6.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.75.pkl
│   ├── zh_en_GA_id_img_feature_dict_0.7.pkl
│   └── zh_en_GA_id_img_feature_dict.pkl
└── UMAEA
    └── save
```

</details>

## 🤝 Cite:
Please condiser citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@inproceedings{DBLP:conf/semweb/ChenGFZCPLCZ23,
  author       = {Zhuo Chen and
                  Lingbing Guo and
                  Yin Fang and
                  Yichi Zhang and
                  Jiaoyan Chen and
                  Jeff Z. Pan and
                  Yangning Li and
                  Huajun Chen and
                  Wen Zhang},
  title        = {Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal
                  Entity Alignment},
  booktitle    = {{ISWC}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14265},
  pages        = {121--139},
  publisher    = {Springer},
  year         = {2023}
}
```

## 💡 Acknowledgement
- Our prior work: [```MEAformer```](https://github.com/zjukg/MEAformer), [```Multi-OpenEA```](https://github.com/THUKElab/Multi-OpenEA)
- We appreciate [```MCLEA```](https://github.com/lzxlin/MCLEA), [```MSNEA```](https://github.com/liyichen-cly/MSNEA), [```EVA```](https://github.com/cambridgeltl/eva), [```MMEA```](https://github.com/liyichen-cly/MMEA) and many other related works for their open-source contributions.
