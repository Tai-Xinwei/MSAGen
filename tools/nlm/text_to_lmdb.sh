#!/usr/bin/bash
set -exuo pipefail

# check if environment variable SAS_TOKEN is set
# if [ -z "${SAS_TOKEN}" ]; then
#     echo "SAS_TOKEN is not set"
#     exit 1
# fi

BLOB_URL='https://msralaphilly2.blob.core.windows.net/ml-la'

FILE_PATH_IN_BLOB_LIST=(
        /home/v-zekunguo/data/yinxia/wu2/shared/SFM/SFM.overall.data/text/valid.patent.v2.txt
        /home/v-zekunguo/data/yinxia/wu2/shared/SFM/SFM.overall.data/text/GPT4-rewrite-nc-cell-medrxiv.txt
        /home/v-zekunguo/data/yinxia/wu2/shared/SFM/SFM.overall.data/text/material_doc_from_ziheng
        /home/v-zekunguo/data/yinxia/wu2/shared/SFM/SFM.overall.data/text/train.patent.v2.txt
        /home/v-zekunguo/data/yinxia/wu2/shared/SFM/SFM.overall.data/text/clinical_trial_intro.txt
        # /home/v-zekunguo/data/yinxia/wu2/shared/SFM/SFM.overall.data/text/SlimPajama_train_sample_300B.txt
        # /blob/v-zequnliu/mix_pretrain/train.c4
        # /blob/v-zequnliu/mix_pretrain/train.pubmedabs
        # /blob/v-kehanwu/data/filtered_data_new/biorxiv.txt
        # /blob/v-kehanwu/data/filtered_data_new/chemrxiv.txt
        # /blob/v-kehanwu/data/filtered_data_new/nature_coms.txt
        # /blob/v-kehanwu/data/filtered_data_new/scientific_reports.txt
        # /blob/v-kehanwu/data/filtered_data_new/beilstein.txt
        # /blob/v-kehanwu/data/filtered_data_new/chemistry_open.txt
        # /blob/v-kehanwu/data/filtered_data_new/rsc.txt
        # /blob/v-kehanwu/data/filtered_data_new/enwiki_filtered.txt
        # /blob/v-kehanwu/data/filtered_data_new/pmc_old.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/text/GPT4-rewrite-nc-cell-medrxiv.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/text/material_doc_from_ziheng
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.pubchem.pureSMILES.randomRoot
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/IUPAC_SMILES_convertion.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.zinc25M-func2mol
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.zinc25M-mol2func
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/Pubchem.fragments.10M
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.Enamine
        # /blob/shufxi/data/scigpt/pistachio_2023Q2_v2_o_smiles/train.txt
        # /blob/lihe/scigpt/data/ur90/train.uniref90.shuf
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/antibody/train.oas
        # /blob/lihe/scigpt/data/dna/sequence/train.DNASeq.20Btoken
        # /blob/lihe/scigpt/data/rna/20.0/train.rnacentral
        # /blob/lihe/scigpt/data/dna/sequence/wrapped_seqs.dna_prot_dna.txt
        # /blob/shufxi/data/scigpt/CrystalLLM/train.txt
        # /blob/v-kehanwu/data/filtered_data_new/train_wrapped_seq.txt
        # /blob/v-kehanwu/data/filtered_data_new/PMC_v1_wrapped.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/protein-text.nonwrap-updated.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/geneannot_protein_desc_v3_nopredict.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/train-text-smiles.txt
        # /blob/shufxi/data/scigpt/text2material/train.txt
        # /blob/shufxi/data/scigpt/materials_project_data/train_x10.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/smallmol_property_train.txt
        # /blob/shufxi/data/scitpt/bindingdb_ec50/train_x2.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/text/clinical_trial_intro.txt
        # /blob/yinxia/wu2/shared/SFM/SFM.overall.data/text/train.patent.v2.txt
)


declare -a TRAIN_FILES_LIST
L=${#FILE_PATH_IN_BLOB_LIST[@]}
S=$1
E=$2
CPUID=$1

echo "there are ${L} files to be processed"
# echo "processing ${S} to ${E}"


for ((ii=S;ii<E;ii+=1)); do
    FILE_PATH_IN_BLOB=${FILE_PATH_IN_BLOB_LIST[${ii}]}
    echo $FILE_PATH_IN_BLOB

    # for FILE_PATH_IN_BLOB in "${FILE_PATH_IN_BLOB_LIST[@]}"; do
    FILE_PATH_IN_LOCAL=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/blob\//\/data\//g')

    # check if local file exists, if not, use azcopy to download
    if [ ! -f "${FILE_PATH_IN_LOCAL}" ]; then
        FILE_RELATIVE_PATH=$(echo "$FILE_PATH_IN_BLOB" | sed 's/\/blob\///g')
        FILE_URL="${BLOB_URL}/${FILE_RELATIVE_PATH}?${SAS_TOKEN}"
        # azcopy copy "${FILE_URL}" "${FILE_PATH_IN_LOCAL}"
    fi

    TRAIN_FILES_LIST+=("${FILE_PATH_IN_LOCAL}")
done

TRAIN_FILES=$(IFS=',' ; echo "${TRAIN_FILES_LIST[*]}")

# TARGET_FOLDER="/home/v-zekunguo/zekun_data/scidata/tmp"
TARGET_FOLDER="/home/v-zekunguo/data/yinxia/wu2/shared/SFM/SFM.overall.data/lmdb"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing train data"

taskset -c $CPUID python -u tools/nlm/text_to_lmdb.py \
--input "${TRAIN_FILES}" \
--output ${TARGET_FOLDER} \
--tokenizer_path /home/v-zekunguo/hai1data/Mixtral-8x7B-v0.1 \
--num_workers 44 \
--seq_len 8192 | tee ${TARGET_FOLDER}/data.log

echo "done"
