cd $HOME

# install tmux
sudo apt install -y tmux

# download azcopy
wget https://azcopyvnext.azureedge.net/releases/release-10.21.2-20231106/azcopy_linux_amd64_10.21.2.tar.gz
tar -xzf azcopy_linux_amd64_10.21.2.tar.gz
rm azcopy_linux_amd64_10.21.2.tar.gz
mv azcopy_linux_amd64_10.21.2 azcopy

SAS_KEY=$1
GRAPHORMER_CKPT_NAME="pretrain56w_pm6oc_512_pm6_oc2_acc2_4UnifiedH_22B_L46_4e4_global_step560000_merged.pt"

# download the enviromment build script
azcopy/azcopy cp "https://hai1data.blob.core.windows.net/mfm/cube/env/$SAS_KEY" . --recursive
# download llama2 model checkpoint
azcopy/azcopy cp "https://hai1data.blob.core.windows.net/mfm/cube/llama2/$SAS_KEY" . --recursive
# download 22b graphormer checkpoint
azcopy/azcopy cp "https://hai1data.blob.core.windows.net/mfm/cube/$GRAPHORMER_CKPT_NAME$SAS_KEY" .
# download data
azcopy/azcopy cp "https://hai1data.blob.core.windows.net/mfm/data/chemical-copilot-special-token-20231129/$SAS_KEY" . --recursive

bash env/build_conda.sh

source /opt/conda/bin/activate
conda activate cubedgx2

bash env/install_cube.sh $CUBE_TOKEN .
bash env/install_sfm.sh $SFM_TOKEN .

mkdir $HOME/ai4sci_22b_70b_stage1
cp $HOME/Fairseq/cube_examples/ai4sci/22b_70b_scripts/run.sh $HOME/ai4sci_22b_70b_stage1
cp $HOME/Fairseq/cube_examples/ai4sci/22b_70b_scripts/run_stage_x.sh $HOME/ai4sci_22b_70b_stage1

mkdir $HOME/ai4sci_22b_70b_stage2
cp $HOME/Fairseq/cube_examples/ai4sci/22b_70b_scripts/run.sh $HOME/ai4sci_22b_70b_stage2
cp $HOME/Fairseq/cube_examples/ai4sci/22b_70b_scripts/run_stage_x.sh $HOME/ai4sci_22b_70b_stage2

mkdir $HOME/ai4sci_22b_70b_stage3
cp $HOME/Fairseq/cube_examples/ai4sci/22b_70b_scripts/run.sh $HOME/ai4sci_22b_70b_stage3
cp $HOME/Fairseq/cube_examples/ai4sci/22b_70b_scripts/run_stage_x.sh $HOME/ai4sci_22b_70b_stage3
