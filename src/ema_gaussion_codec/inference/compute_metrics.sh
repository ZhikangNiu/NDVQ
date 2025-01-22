#!/usr/bin/env bash

stage=2
stop_stage=6
ref_dir=$1
gen_dir=$2
bw=$3

echo ${ref_dir}
echo ${gen_dir}

if [ $stage -le 1 ] && [ "${stop_stage}" -ge 2 ];then
  echo "Compute STOI"
  python metrics/compute_stoi.py \
    -r ${ref_dir} \
    -d ${gen_dir} \
    -b ${bw}
fi

if [ $stage -le 2 ] && [ "${stop_stage}" -ge 3 ];then
  echo "Compute PESQ"
  python metrics/compute_pesq.py \
    -r ${ref_dir} \
    -d ${gen_dir} \
    -b ${bw}
fi

if [ $stage -le 3 ] && [ "${stop_stage}" -ge 4 ];then
  echo "Compute Distance"
  python metrics/compute_distance.py \
    -r ${ref_dir} \
    -d ${gen_dir} \
    -b ${bw}
fi

if [ $stage -le 4 ] && [ "${stop_stage}" -ge 5 ];then
  echo "Compute SISDR"
  python metrics/compute_sisdr.py \
    -r ${ref_dir} \
    -d ${gen_dir} \
    -b ${bw}

fi

if [ $stage -le 5 ] && [ "${stop_stage}" -ge 6 ];then
  echo "Compute VISQOL"
  python metrics/compute_visqol.py \
    -r ${ref_dir} \
    -d ${gen_dir} \
    -b ${bw}

fi

if [ $stage -le 6 ] && [ "${stop_stage}" -ge 7 ];then
  echo "Compute MOS"
  python metrics/compute_speechmos.py \
  -d ${gen_dir} \
  -b ${bw}

fi