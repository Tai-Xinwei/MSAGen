#!/usr/bin/bash
set -xeuo pipefail

sfm_url='git@ssh.dev.azure.com:v3/AI4ScienceSFM/SFM_framework/SFM_framework'
sfm_home=/tmp/smf
feyman_home="/tmp/feyman"
feyman_sfm_home="$feyman_home/projects/SFM/experimental/SFM_framework"


# clone the SFM_framework
if [ -d "$sfm_home" ]; then
    rm -rf $sfm_home
fi
git clone $sfm_url $sfm_home --depth 1


# clone the feyman
if [ -d "$feyman_home" ]; then
    rm -rf $feyman_home
fi
gh repo clone msr-ai4science/feynman $feyman_home

cd $feyman_home

# name the branch as sfm_date_time
git checkout -b sfm_$(date +%Y%m%d_%H%M%S)

# copy the files
rm -rf $feyman_sfm_home
cp -r $sfm_home $feyman_sfm_home

# remove unnecessary files

for p in .git .gitignore  azure-pipelines.yml  tools/sync.sh; do
    if [[ -f $feyman_sfm_home/$p ]] || [[ -d $feyman_sfm_home/$p ]] ; then
        rm -rf $feyman_sfm_home/$p
    fi
done

find $feyman_sfm_home -size +1M -exec rm -rf {} \;

git add $feyman_sfm_home
git commit -m "feat(SFM): update SFM_framework at $(date +%Y%m%d_%H%M%S)"
git push origin sfm_$(date +%Y%m%d_%H%M%S)
