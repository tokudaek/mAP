#!/bin/bash

R=/home/ibaraujo/cvpr19-dataset/detection/
#DIRS=( fasterrcnn_ddn fasterrcnn_deraindrop fasterrcnn_did-mdn fasterrcnn_idcgan fasterrcnn_jorder fasterrcnn_raw ssd_ddn ssd_deraindrop ssd_did-mdn ssd_idcgan ssd_jorder ssd_raw )
#DIRS=( ssd_raw ssd_jorder ssd_ddn ssd_idcgan ssd_did-mdn ssd_deraindrop )
DIRS=( fasterrcnn_raw fasterrcnn_jorder fasterrcnn_ddn fasterrcnn_idcgan fasterrcnn_did-mdn fasterrcnn_deraindrop )
ANN=/home/keiji/temp/rainy_set_annotations_motorbike/
#ANN=/home/keiji/temp/rainy_set_annotations/

for D in "${DIRS[@]}"; do
	echo python main.py  --preddir=${R}/$D \
		--gnddir=$ANN \
		--outfile=$R/${D}.txt
done

