#!/bin/bash

R=$HOME/results/derain/derain_benchmark/
ROOT=$R/detections/
METHS=( maskrcnn ssd )
DETS=( aodnet_paired aodnet_unpaired dcgan_paired deraindrop_paired deraindrop_unpaired didmdn_paired didmdn_unpaired idcgan_unpaired rainy_paired rainy_unpaired rescan_paired rescan_unpaired sunny_paired sunny_unpaired ugsm_paired ugsm_unpaired )

METH="${METHS[0]}"

for DET in "${DETS[@]}"; do
	if [[ $DET = *"unpaired"* ]]; then P='unpaired'; else P='paired'; fi
	if [[ $DET = *"sunny"* ]]; then W='sunny'; else W='rainy'; fi 

	python main.py  --preddir=$ROOT/$METH/$DET/ \
		--gnddir=$R/real/${W}_${P}_annot/ \
		--outfile=$ROOT/${METH}_${DET}.txt
done

