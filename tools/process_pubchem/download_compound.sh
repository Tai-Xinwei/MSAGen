#!/usr/bin/env bash

url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"

for i in {0..336}; do
  t=$(($i*500000))
  s=$(($t+1))
  e=$(($t+500000))

  s=$(printf "%09d" $s)
  e=$(printf "%09d" $e)
  wget $url"Compound_"$s"_"$e".sdf.gz"
done
