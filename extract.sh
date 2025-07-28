#!/usr/bin/env bash

IMG_DIR="/data2/npl/ViInforgraphic/data/images"
MODEL_FILE="/data2/npl/ViInforgraphic/workspace/feature_extraction/frcnn/detectron_model_2.pth"
CONFIG_FILE="/data2/npl/ViInforgraphic/workspace/feature_extraction/frcnn/detectron_model_2.yaml"
BATCH_SIZE=32
NUM_FEATURES=100
OUTPUT_FOLDER="/data2/npl/ViInfographic/data/faster_rcnn"
FEATURE_NAME="fc6"
CONF_THRESH=0.5
DEVICE="cuda:2"

python /data2/npl/ViInforgraphic/workspace/feature_extraction/frcnn/FasterRCNN-Feature-Extraction/extract_features.py --model_file "$MODEL_FILE" --config_file "$CONFIG_FILE" --batch_size $BATCH_SIZE --num_features $NUM_FEATURES --output_folder "$OUTPUT_FOLDER" --image_dir "$IMG_DIR" --feature_name $FEATURE_NAME --confidence_threshold $CONF_THRESH --device $DEVICE --background 

