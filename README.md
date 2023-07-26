# UMAEA
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/UMAEA/blob/main/LICENSE)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ISWC2023](https://img.shields.io/badge/ISWC-2023-%23bd9f65?labelColor=%23bea066&color=%23ffffff)](https://iswc2023.semanticweb.org/)

- [*Rethinking Uncertainly Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment*]()
>In the face of modality incompleteness, some models succumb to overfitting the modality noise, and exhibit performance oscillations or declines at high modality missing rates. This indicates that the inclusion of additional multi-modal data can sometimes **adversely affect EA**. To address these challenges, we introduces **`UMAEA`**, a robust multi-modal entity alignment approach designed to tackle **uncertainly missing and ambiguous visual modalities**.

<div align="center">
    <img src="https://github.com/zjukg/UMAEA/blob/main/IMG/case.jpg" width="70%" height="auto" />
</div>

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
```bash
# Command Details:
# bash file / GPU / Dataset / data split / R_{sa} / R_{img}
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

❗Tips: you can open the `run_umaea_X.sh` file for parameter or training target modification.

## 📚 Dataset (MMEA-UMVM)
>To create our `MMEA-UMVM` (uncertainly missing visual modality) datasets, we perform **random image dropping** on MMEA datasets. Specifically, we randomly discard entity images to achieve varying degrees of visual modality missing, ranging from 0.05 to the maximum $R_{img}$ of the raw datasets with a step of 0.05 or 0.1 (The raw images are available at [```Multi-OpenEA```](https://github.com/THUKElab/Multi-OpenEA)). Finally, we get a total number of 97 data split as follow:

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

❗NOTE: Download from [GoogleDrive](https://drive.google.com/file/d/1TDESVvXh5eq2aW50qGuqqNajy5Mkc6Nw/view?usp=sharing) (6.09G) and unzip it to make those files **satisfy the following file hierarchy**:
```
ROOT
├── data
│   └── mmkg
└── code
    └── UMAEA
```

#### Code Path
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




#### Data Path
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

## 🤝 Cite:
Please condiser citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@inproceedings{chen2023rethinking,
  author       = {Zhuo Chen and
                  Lingbing Guo and
                  Yin Fang and
                  Yichi Zhang and
                  Jiaoyan Chen and
                  Wen Zhang and
                  Jeff Z. Pan and
                  Yangning Li and
                  Huajun Chen},
  title        = {Rethinking Uncertain Missing and Ambiguous Visual Modality in Multi-Modal Entity Alignment},
  booktitle    = {{ISWC}},
  series       = {Lecture Notes in Computer Science},
  publisher    = {Springer},
  year         = {2023}
}
```

## 💡 Acknowledgement
- Our prior work: [```MEAformer```](https://github.com/zjukg/MEAformer), [```Multi-OpenEA```](https://github.com/THUKElab/Multi-OpenEA)
- We appreciate [```MCLEA```](https://github.com/lzxlin/MCLEA), [```MSNEA```](https://github.com/liyichen-cly/MSNEA), [```EVA```](https://github.com/cambridgeltl/eva), [```MMEA```](https://github.com/liyichen-cly/MMEA) and many other related works for their open-source contributions.
