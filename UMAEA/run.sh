# Command Details:
# bash file / GPU / Dataset / data split / R_{sa} / R_{img}
# Begin:
# ---------- R_{img} = 0.4 & iter. & w/o CMMI ----------
bash run_umaea_00.sh 0 OEA_D_W_15K_V1 norm 0.2 0.4
bash run_umaea_00.sh 0 OEA_D_W_15K_V2 norm 0.2 0.4
bash run_umaea_00.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.4
bash run_umaea_00.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.4
bash run_umaea_00.sh 0 DBP15K fr_en 0.3 0.4
bash run_umaea_00.sh 0 DBP15K ja_en 0.3 0.4
bash run_umaea_00.sh 0 DBP15K zh_en 0.3 0.4
# ---------- R_{img} = 0.6 & non-iter. & w/o CMMI ----------
bash run_umaea_0.sh 0 OEA_D_W_15K_V1 norm 0.2 0.6
bash run_umaea_0.sh 0 OEA_D_W_15K_V2 norm 0.2 0.6
bash run_umaea_0.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.6
bash run_umaea_0.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.6
bash run_umaea_0.sh 0 DBP15K fr_en 0.3 0.6
bash run_umaea_0.sh 0 DBP15K ja_en 0.3 0.6
bash run_umaea_0.sh 0 DBP15K zh_en 0.3 0.6
# --------- R_{img} = 0.1 & non-iter. & w/ CMMI ---------
bash run_umaea_012.sh 0 OEA_D_W_15K_V1 norm 0.2 0.1
bash run_umaea_012.sh 0 OEA_D_W_15K_V2 norm 0.2 0.1
bash run_umaea_012.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.1
bash run_umaea_012.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.1
bash run_umaea_012.sh 0 DBP15K fr_en 0.3 0.1
bash run_umaea_012.sh 0 DBP15K ja_en 0.3 0.1
bash run_umaea_012.sh 0 DBP15K zh_en 0.3 0.1
# --------- R_{img} = 0.2 & iter. & w/ CMMI ---------
bash run_umaea_012012.sh 0 OEA_D_W_15K_V1 norm 0.2 0.2
bash run_umaea_012012.sh 0 OEA_D_W_15K_V2 norm 0.2 0.2
bash run_umaea_012012.sh 0 OEA_EN_FR_15K_V1 norm 0.2 0.2
bash run_umaea_012012.sh 0 OEA_EN_DE_15K_V1 norm 0.2 0.2
bash run_umaea_012012.sh 0 DBP15K fr_en 0.3 0.2
bash run_umaea_012012.sh 0 DBP15K ja_en 0.3 0.2
bash run_umaea_012012.sh 0 DBP15K zh_en 0.3 0.2