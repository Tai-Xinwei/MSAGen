#!/bin/bash
# Folders
[ -z "${ckpt_name}" ] && ckpt_name='1b_dialogue_1v1_bs2048_steps_20000'
# 1b_dialogue_1v1_bs2048_steps_18382
# 1b_dialogue_1v1_bs2048_steps_20000
# 1b_dialogue_3v1_bs2048_steps_14816
# 1b_dialogue_3v1_bs2048_steps_20000
# 1b1t_bs256_steps_105255
[ -z "${results_dir}" ] && results_dir="/home/v-zekunguo/nlm/zekun/instruct/base1b/instruct_task_20240807/$ckpt_name/all"
# [ -z "${results_dir}" ] && results_dir='/home/v-zekunguo/nlm/zekun/instruct/base1b/instruct_task_20240807/1b1t_bs256_steps_116950/all'
# # [ -z "${input_dir}" ] && input_dir='/home/v-yinzhezhou/new_branch_SFM/SFM_framework/eval_testing_data/evaluate_small_molecule/input_data'
[ -z "${output_dir}" ] && output_dir='/home/v-zekunguo/nlm/zekun/instruct/inst_result/instruct_task_20240807'



# [ -z "${results_dir}" ] && results_dir='/home/v-zekunguo/nlm/zekun/instruct/base8b/instruct_task_new/SFMMolInstruct/all'
# # [ -z "${input_dir}" ] && input_dir='/home/v-yinzhezhou/new_branch_SFM/SFM_framework/eval_testing_data/evaluate_small_molecule/input_data'
# [ -z "${output_dir}" ] && output_dir='/home/v-zekunguo/nlm/zekun/instruct/base8b/instruct_task_new/SFMMolInstruct/all'

# [ -z "${results_dir}" ] && results_dir='/home/v-zekunguo/logs/base1b/alpha0.3_doubleProtein_local'
[ -z "${input_dir}" ] && input_dir='/home/v-zekunguo/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617.test'
# [ -z "${output_dir}" ] && output_dir='/home/v-zekunguo/logs/base1b/alpha0.3_doubleProtein_local'
# Files
[ -z "${bbbp_pkl}" ] && bbbp_pkl='test.instruct.predict_bbbp.tsv.response.pkl'
[ -z "${bbbp_score_pkl}" ] && bbbp_score_pkl='test.instruct.predict_bbbp.tsv.score.pkl'
[ -z "${herg_pkl}" ] && herg_pkl='None'
[ -z "${i2s_s_txt}" ] && i2s_s_txt='iupac_smiles_translation/test.raw.i2s_s.txt'
[ -z "${i2s_i_pkl}" ] && i2s_i_pkl='test.new.i2s_i.txt.response.pkl'
[ -z "${s2i_i_txt}" ] && s2i_i_txt='iupac_smiles_translation/test.raw.s2i_i.txt'
[ -z "${s2i_s_pkl}" ] && s2i_s_pkl='test.new.s2i_s.txt.response.pkl'
[ -z "${desc2mol_pkl}" ] && desc2mol_pkl='test.desc2mol.tsv.response.pkl'
[ -z "${molinstruct_pkl}" ] && molinstruct_pkl='test.molinstruct.reagent_prediction.tsv.response.pkl'
[ -z "${mol2desc_pkl}" ] && mol2desc_pkl='test.mol2desc.tsv.response.pkl'
[ -z "${bace_tsv}"] && bace_tsv='test.instruct.predict_bace.tsv '
[ -z "${bace_pkl}"] && bace_pkl='test.instruct.predict_bace.tsv.response.pkl'
[ -z "${bace_score_pkl}"] && bace_score_pkl='test.instruct.predict_bace.tsv.score.pkl'
[ -z "${class_pkl}" ] && class_pkl='sfmdata.prot.test.sampled30.tsv.response.pkl,kdr-hi_prediction_test_1.txt.response.pkl,drd2-hi_prediction_test_1.txt.response.pkl,hiv-hi_prediction_test_1.txt.response.pkl,sol-hi_prediction_test_1.txt.response.pkl,test.instruct.predict_bbbp.tsv.response.pkl,test.instruct.predict_bace.tsv.response.pkl,sfmdata.prot.test.sampled30.tsv.response.pkl,Core_Promoter_detection_test.tsv.response.pkl,Promoter_detection_test.tsv.response.pkl,Transcription_factor_binding_site_prediction_0_test.tsv.response.pkl,Transcription_factor_binding_site_prediction_1_test.tsv.response.pkl,Transcription_factor_binding_site_prediction_2_test.tsv.response.pkl,Transcription_factor_binding_site_prediction_3_test.tsv.response.pkl,Transcription_factor_binding_site_prediction_4_test.tsv.response.pkl'
[ -z "${regress_pkl}" ] && regress_pkl='test.BindngDB_pIC50_reg.500.tsv.response.pkl,test.BindngDB_pKd_reg.500.tsv.response.pkl,test.BindngDB_pKi_reg.500.tsv.response.pkl,drd2-lo_prediction_test_1.txt.response.pkl,kcnh2-lo_prediction_test_1.txt.response.pkl,kdr-lo_prediction_test_1.txt.response.pkl,human_enhancer_K562_test.tsv.response.pkl,yeast_prom_complex_test.tsv.response.pkl'
[ -z "${retro_pkl}" ] && retro_pkl='test.uspto50k.retro.osmi.tsv.response.pkl'
[ -z "${absolute_correct_pkl}" ] && absolute_correct_pkl='val.grna.improve.classification.filter.response.pkl,test.uspto50k.reaction.osmi.tsv.response.pkl,test.molinstruct.reagent_prediction.tsv.response.pkl'
[ -z "${target_to_drug_pkl}" ] && target_to_drug_pkl='test.SFMinstruct.t2d.tsv.response.pkl,test.SFMinstruct.t2f.tsv.response.pkl'
[ -z "${antibody_pkl}" ] && antibody_pkl='test.antibody.design.tsv.response.pkl,test.antigen_antibody.design.tsv.response.pkl,test.antigen_to_full_antibody.design.tsv.response.pkl'
[ -z "${drug_assist_folder}" ] && drug_assist_folder='/home/v-zekunguo/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617/Drug_Assist'
[ -z "${drug_assist_pkl}" ] && drug_assist_pkl='test.drug.donor.tsv.response.pkl,test.drug.logp.tsv.response.pkl,test.drug.qed.tsv.response.pkl'
[ -z "${grna_filter_pkl}" ] && grna_filter_pkl='val.grna.filter.response.pkl'
[ -z "${result_pkl}" ] && result_pkl='/home/v-zekunguo/nlm/zekun/instruct/inst_result/instruct_task_20240807'
[ -z "${save_excel_path}" ] && save_excel_path='/home/v-zekunguo/nlm/zekun/instruct/inst_result/instruct_task_20240807'

[ -z "${gen_cyp_pkl}"] && gen_cyp_pkl='task_cyp450_cyp1a2_res.pkl,task_cyp450_cyp2c19_res.pkl,task_cyp450_cyp2c9_res.pkl,task_cyp450_cyp2d6_res.pkl,task_cyp450_cyp3a4_res.pkl'
[ -z "${base_gen_cyp_path}"] && base_gen_cyp_path='/home/v-zekunguo/nlm/zekun/instruct/base1b/instruct_task_20240807/wang/ADMET.v2/result'
[ -z "${gen_bbbp_pkl}"] && gen_bbbp_pkl='/home/v-zekunguo/nlm/zekun/instruct/base1b/instruct_task_20240807/wang/bbbp.v2/result/task_bbbp_res.pkl'
[ -z "${protein2desc_pkl}"] && protein2desc_pkl='catalytic_activity_test.tsv.response.pkl,domain_motif_test.tsv.response.pkl,general_function_test.tsv.response.pkl,protein_function_test.tsv.response.pkl,instruct_gene_annot_031824_test_no_predict.tsv.response.pkl'

[ -z "${merge_result}" ] && merge_result='True'
if [[ "${merge_result}" == "True" ]]; then
  merge_result="--merge_result"
else
  merge_result=""
fi
# Launcher
python3 sfm/tasks/nlm/eval/evaluate_sfm.py \
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
    --bace_score_pkl $bace_score_pkl \
    --regress_pkl $regress_pkl \
    --class_pkl $class_pkl \
    --retro_pkl $retro_pkl \
    --absolute_correct_pkl $absolute_correct_pkl \
    --target_to_drug_pkl $target_to_drug_pkl \
    --antibody_pkl $antibody_pkl \
    --drug_assist_folder $drug_assist_folder \
    --drug_assist_pkl $drug_assist_pkl \
    --grna_filter_pkl $grna_filter_pkl \
    --result_pkl $result_pkl \
    --ckpt_name $ckpt_name \
    --gen_cyp_pkl $gen_cyp_pkl \
    --base_gen_cyp_path $base_gen_cyp_path \
    --gen_bbbp_pkl $gen_bbbp_pkl \
    --protein2desc_pkl $protein2desc_pkl \
    --save_excel_path $save_excel_path \
    $merge_result \
