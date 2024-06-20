#!/bin/bash
# Folders
[ -z "${results_dir}" ] && results_dir='/home/yeqibai/workspace/SFM_NLM_resources/all_infer_results/8x7b_global_step33216_infer_results'
[ -z "${input_dir}" ] && input_dir='/home/yeqibai/mount/ml_la/yeqibai/warehouse/nlm_data/instruct/molecules_test/'
[ -z "${output_dir}" ] && output_dir='/home/yeqibai/workspace/SFM_NLM_resources/all_eval_results/8x7b_global_step33216_infer_results'


# Files
[ -z "${bbbp_pkl}" ] && bbbp_pkl='test.bbbp.instruct.tsv.response.pkl'
[ -z "${bbbp_score_pkl}" ] && bbbp_score_pkl='test.bbbp.instruct.tsv.score.pkl'
[ -z "${herg_pkl}" ] && herg_pkl='test.hERG.response.pkl'
[ -z "${i2s_s_txt}" ] && i2s_s_txt='test.raw.i2s_s.txt'
[ -z "${i2s_i_pkl}" ] && i2s_i_pkl='i2s_i.txt.response.pkl'
[ -z "${s2i_i_txt}" ] && s2i_i_txt='test.raw.s2i_i.txt'
[ -z "${s2i_s_pkl}" ] && s2i_s_pkl='s2i_s.txt.response.pkl'
[ -z "${desc2mol_pkl}" ] && desc2mol_pkl='test.desc2mol.response.pkl'
[ -z "${molinstruct_pkl}" ] && molinstruct_pkl='test.molinstruct.reaction.response.pkl'
[ -z "${mol2desc_pkl}" ] && mol2desc_pkl='test.mol2desc.response.pkl'
[ -z "${bace_tsv}"] && bace_tsv='test.bace.instruct.tsv'
[ -z "${bace_pkl}"] && bace_pkl='test.bace.instruct.tsv.response.pkl'
[ -z "${bace_score_pkl}"] && bace_score_pkl='test.bace.instruct.tsv.score.pkl'

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
