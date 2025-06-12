#!/usr/bin/env bash

IMG_DIR="/data/npl/ViInfographicCaps/images/images"
MODEL_FILE="/data/npl/ViInfographicCaps/workspace/feature_extraction/frcnn/detectron_model_2.pth"
CONFIG_FILE="/data/npl/ViInfographicCaps/workspace/feature_extraction/frcnn/detectron_model_2.yaml"
BATCH_SIZE=64
NUM_FEATURES=100
OUTPUT_FOLDER="/data/npl/ViInfographicCaps/features/faster_rcnn"
FEATURE_NAME="fc6"
CONF_THRESH=0.5
DEVICE="cuda:5"

python /data/npl/ViInfographicCaps/workspace/feature_extraction/frcnn/extract.py --model_file "$MODEL_FILE" --config_file "$CONFIG_FILE" --batch_size $BATCH_SIZE --num_features $NUM_FEATURES --output_folder "$OUTPUT_FOLDER" --image_dir "$IMG_DIR" --feature_name $FEATURE_NAME --confidence_threshold $CONF_THRESH --device $DEVICE --background

