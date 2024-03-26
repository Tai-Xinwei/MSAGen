#!/usr/bin/bash
set -exuo pipefail

TRAIN_FILES="/blob/v-zequnliu/mix_pretrain/c4,
        /blob/v-zequnliu/mix_pretrain/train.pubmed_15M_title_abs.detok,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.pubchem.30M.can.tok.marked.pended.new.txt,
        /blob/shufxi/data/scigpt/ur50/uniref50_2023_05.shorten.train.taged.seqs,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/antibody/oas-5m-sampled-formatted.txt,
        /blob/shufxi/data/scigpt/CrystalLLM/train.txt,
        /blob/shufxi/data/scigpt/pistachio_2023Q2_v2_o_smiles/train.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/wrapped_data/train_wrapped_seq.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/wrapped_data/PMC_v1_wrapped.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/IUPAC_SMILES_convertion.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/protein-text.nonwrap-updated.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/geneannot_protein_desc_v2.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/train-text-smiles.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/zinc25M-func2mol,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/zinc25M-mol2func,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/Pubchem.fragments.10M,
        /blob/shufxi/data/scigpt/text2material/train.txt,
        /blob/shufxi/data/scigpt/materials_project_data/train_x10.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/smallmol_property_train.txt,
        /blob/shufxi/data/scitpt/bindingdb_ec50/train_x2.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/DNA/dnaseq.9Btoken.txt,
        /blob/lihe/scigpt/data/dna/sequence/wrapped_seqs.dna_prot_dna.txt,
        /blob/shufxi/data/scigpt/rna/RNA_center_uniq.processed.txt"


TARGET_FOLDER="/home/shufxi/scigptdata/v3"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing train data"
if [ ! -f "${TARGET_FOLDER}/train.npy" ]; then
        python -u tools/scigpt/make_binary.py \
        --input "${TRAIN_FILES}" \
        --output ${TARGET_FOLDER}/train.npy \
        --tokenizer_path /hai1/ds_dataset/llama2/llama-2-7b \
        --num_workers "${WORKERS}" \
        --seq_len 4096
fi
echo "done"
