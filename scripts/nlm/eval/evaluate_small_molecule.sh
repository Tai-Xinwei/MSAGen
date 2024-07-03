#!/bin/bash
# Folders
[ -z "${results_dir}" ] && results_dir='/home/t-kaiyuangao/workspace/proj_logs/nlm_inst/inst_0621_bsz256_lr2e5_0624_step89920'
[ -z "${input_dir}" ] && input_dir='/home/t-kaiyuangao/ml-container/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617.test'
[ -z "${output_dir}" ] && output_dir='/home/t-kaiyuangao/workspace/proj_logs/nlm_inst/inst_0621_bsz256_lr2e5_0624_step89920/eval_results/'


# Files
[ -z "${bbbp_pkl}" ] && bbbp_pkl='test.instruct.predict_bbbp.tsv.response.pkl'
[ -z "${bbbp_score_pkl}" ] && bbbp_score_pkl='test.instruct.predict_bbbp.tsv.score.pkl'
[ -z "${herg_pkl}" ] && herg_pkl='None'
[ -z "${i2s_s_txt}" ] && i2s_s_txt='iupac_smiles_translation/test.raw.i2s_s.txt'
[ -z "${i2s_i_pkl}" ] && i2s_i_pkl='test.new.i2s_i.txt.response.pkl'
[ -z "${s2i_i_txt}" ] && s2i_i_txt='iupac_smiles_translation/test.raw.s2i_i.txt'
[ -z "${s2i_s_pkl}" ] && s2i_s_pkl='test.new.s2i_s.txt.response.pkl'
[ -z "${desc2mol_pkl}" ] && desc2mol_pkl='test.desc2mol.tsv.response.pkl'
[ -z "${molinstruct_pkl}" ] && molinstruct_pkl='test.molinstruct.reagent_prediction.tsv.response.pkl '
[ -z "${mol2desc_pkl}" ] && mol2desc_pkl='test.mol2desc.tsv.response.pkl'
[ -z "${bace_tsv}"] && bace_tsv='test.instruct.predict_bace.tsv '
[ -z "${bace_pkl}"] && bace_pkl='test.instruct.predict_bace.tsv.response.pkl'
[ -z "${bace_score_pkl}"] && bace_score_pkl='test.instruct.predict_bace.tsv.score.pkl'

# Launcher
python3 sfm/tasks/nlm/eval/evaluate_small_molecule.py \
    --results_dir $results_dir \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --bbbp_pkl $bbbp_pkl \
    --bbbp_score_pkl $bbbp_score_pkl \
    --herg_pkl $herg_pkl \
    --i2s_s_txt $i2s_s_txt \
    --i2s_i_pkl $i2s_i_pkl \
    --s2i_i_txt $s2i_i_txt \
    --s2i_s_pkl $s2i_s_pkl \
    --desc2mol_pkl $desc2mol_pkl \
    --molinstruct_pkl $molinstruct_pkl \
    --mol2desc_pkl $mol2desc_pkl \
    --bace_tsv $bace_tsv \
    --bace_pkl $bace_pkl \
    --bace_score_pkl $bace_score_pkl
