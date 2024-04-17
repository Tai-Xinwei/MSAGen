#!/usr/bin/bash
set -exuo pipefail

TRAIN_FILES="/blob/v-zequnliu/mix_pretrain/c4,
        /blob/v-zequnliu/mix_pretrain/train.pubmed_15M_title_abs.detok,
        /blob/v-kehanwu/data/filtered_data/biorxiv.txt,
        /blob/v-kehanwu/data/filtered_data/arxiv.txt,
        /blob/v-kehanwu/data/filtered_data/chemrxiv.txt,
        /blob/v-kehanwu/data/filtered_data/nature_coms.txt,
        /blob/v-kehanwu/data/filtered_data/scientific_reports.txt,
        /blob/v-kehanwu/data/filtered_data/beilstein.txt,
        /blob/v-kehanwu/data/filtered_data/chemistry_open.txt,
        /blob/v-kehanwu/data/filtered_data/rsc.txt,
        /blob/v-kehanwu/data/filtered_data/enwiki_filtered.txt,
        /blob/v-kehanwu/data/filtered_data/pmc_old.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/text/GPT4-rewrite-nc-cell-medrxiv.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/text/material_doc_from_ziheng,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.pubchem.30M.can.tok.marked.pended.new.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.pubchem.GridSampleBasedCluster.mol.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/IUPAC_SMILES_convertion.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.zinc25M-func2mol,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.zinc25M-mol2func,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/Pubchem.fragments.10M,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/molecule/train.Enamine,
        /blob/shufxi/data/scigpt/pistachio_2023Q2_v2_o_smiles/train.txt,
	/blob/lihe/scigpt/data/ur90/uniref90_2024_02.pended.seq.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/antibody/oas-5m-sampled-formatted.txt,
        /blob/lihe/scigpt/data/dna/sequence/DNAseq.20Btoken.txt,
        /blob/lihe/scigpt/data/rna/20.0/rnacentral_species_specific_ids.pended.seq.txt,
        /blob/lihe/scigpt/data/dna/sequence/wrapped_seqs.dna_prot_dna.txt,
        /blob/shufxi/data/scigpt/CrystalLLM/train.txt,
        /blob/v-kehanwu/data/filtered_data/train_wrapped_seq.txt,
        /blob/v-kehanwu/data/filtered_data/PMC_v1_wrapped.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/protein-text.nonwrap-updated.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/geneannot_protein_desc_v3.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/train-text-smiles.txt,
        /blob/shufxi/data/scigpt/text2material/train.txt,
        /blob/shufxi/data/scigpt/materials_project_data/train_x10.txt,
        /blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/smallmol_property_train.txt,
        /blob/shufxi/data/scitpt/bindingdb_ec50/train_x2.txt,"

TARGET_FOLDER="/hai1/shufxi/data/scigpt/v4"
mkdir -p "${TARGET_FOLDER}"
WORKERS=$(nproc)


echo "processing train data"
if [ ! -f "${TARGET_FOLDER}/train.npy" ]; then
        python -u tools/nlm/make_binary.py \
        --input "${TRAIN_FILES}" \
        --output ${TARGET_FOLDER}/train.npy \
        --tokenizer_path /hai1/shufxi/Mixtral-8x7B-v0.1 \
        --num_workers "${WORKERS}" \
        --seq_len 8192 | tee ${TARGET_FOLDER}/data.log
fi
echo "done"
