AZCOPYDIR=~
i=$1
SAS=$2

DOWNLOAD=~/download
SAVEDIR=~/$i
mkdir -p $DOWNLOAD $SAVEDIR

wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar xvzf azcopy.tar.gz -C $AZCOPYDIR && mv $AZCOPYDIR/azcopy_linux_amd64_*/azcopy $AZCOPYDIR/azcopy && rm -rf $AZCOPYDIR/azcopy_linux_amd64_* azcopy.tar.gz

echo "Processing $i shard"
$AZCOPYDIR/azcopy cp 'https://sfmdata.blob.core.windows.net/protein/AFDBv4-1K/'"$i$SAS" $DOWNLOAD --recursive
ret=$(head -1 $DOWNLOAD/$i/gcloud_log.txt)
if [ "$ret" = "RETURN CODE: 0" ]; then
    echo "Download check pass" | tee -a $SAVEDIR/$i.log
else
    echo "Download check failed" | tee -a $SAVEDIR/$i.log
    break
fi
# use GNU parallel to speed up the untar
find $DOWNLOAD/$i -name "*.tar" | parallel -j -1 "tar -xf {} -C $DOWNLOAD/$i/ && rm {}"
python tools/protein_data_process/structure2lmdb.py processafdb $DOWNLOAD/$i $SAVEDIR/$i.lmdb --glob '**/*.cif.gz' --num-workers -1 2>&1 | tee -a $SAVEDIR/$i.log
rm -rf $DOWNLOAD/$i

$AZCOPYDIR/azcopy cp $SAVEDIR 'https://sfmdata.blob.core.windows.net/protein/AFDBv4-processed/'"$SAS" --recursive
