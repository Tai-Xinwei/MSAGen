ulimit -c unlimited

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

wget 'https://aka.ms/downloadazcopy-v10-linux' -O /tmp/azcopy.tar.gz
tar -xf /tmp/azcopy.tar.gz -C /tmp
# find the folder in /tmp and starts with azcopy_linux_amd64
azcopy_path=$(find /tmp -maxdepth 1 -type d -name 'azcopy_linux_amd64*')
# # $azcopy_path/azcopy copy ... ... --recursive
mkdir -p /tmp/psmdata/
