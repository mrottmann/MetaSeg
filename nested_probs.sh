#!/bin/bash
#
#	derived from deeplab/local_test.sh
# 	usage: sh ./nested_probs.sh
#


# Exit immediately if a command exits with a non-zero status.
set -e

METASEG_FOLDER="/home/rottmann/metaseg"

# adjust GPU id
export CUDA_VISIBLE_DEVICES=1

# path to tensorflow slim
export PYTHONPATH=$PYTHONPATH:`pwd`:"${METASEG_FOLDER}":"${METASEG_FOLDER}/slim"

# ----------------------------------------------------------------------
# adjustable parameters
CALC_CROPS=1  # construct the hierarchy of nested crops for each image
DATA_TO_TF=1  # convert all cropped images to tensorflow format
CALC_PROBS=1  # infer probability distributions for each crop
MERGE_PROBS=1 # for each image, merge the the infered probability distributions
              # to a common one and compute all heatmaps

CROP_DIST=20  # crop CROP_DIST pixels on left and right as well as
              # CROP_DIST/2 pixels at top and bottom
NUM_CROPS=16  # number of nested crops
# ----------------------------------------------------------------------
              
# parameters selecting dataset and model
DATASET="cityscapes" # {"cityscapes","pascal_voc_seg","ade20k"}
MODEL_VARIANT="mobilenet_v2" # {"mobilenet_v2" (fast model),"xception_65" (best model)}
MODEL_VARIANT_SUB="coarse" # {"fine","coarse"} 
# fine: outputstride=8 (+ matching atrous rates), multi-scale eval / coarse: outputstride=16 (+ matching atrous rates), single-scale eval / selection details coded in vis_stat.py
VIS_SPLIT="val" # {train,val,test}

# general parameters
WORK_DIR="${METASEG_FOLDER}/deeplab"
PARENT_WORK_DIR="${WORK_DIR}/.."
DATASETS_DIR="${METASEG_FOLDER}/deeplab/datasets"
CHECKPOINTS_DIR="${WORK_DIR}/../models_checkpoints"

# ----------------------------------------------------------------------
# leave as is
BATCH_SIZE=1
NUM_BATCHES=500
CREATE_IMAGES=0 # if 1 create prediction-related images
SAVE_PROBS=1 # if 1 save probabilities
COPY_GROUND_TRUTH=1 # only for CREATE_IMAGES=1
MAKE_OVERLAYS=1 # only for CREATE_IMAGES=1
# ----------------------------------------------------------------------
# choice of networks
if [ "${DATASET}" = "cityscapes" ]
then
	DATASET_DIR="cityscapes"
	GT_DIR="${DATASETS_DIR}/${DATASET_DIR}/gtFine"
	if [ "${MODEL_VARIANT}" = "mobilenet_v2" ]
	then
		CHECKPOINT_DIR="deeplabv3_mnv2_cityscapes_train"
		MODEL_NAME="mn.sscl.os16"
	else
		CHECKPOINT_DIR="deeplabv3_cityscapes_train"
		MODEL_NAME="xc.mscl.os8"
	fi
	COLORMAP_TYPE="cityscapes"
	VIS_CROP_SIZE_H=1024
	VIS_CROP_SIZE_W=2048
fi

ORIG_SIZE_H=1024
ORIG_SIZE_W=2048



for CROP_NUMBER in $(seq 0 1 $((NUM_CROPS-1)))

# for CROP_NUMBER in $(seq 1 1 1)
do
  echo "cropping number ${CROP_NUMBER} ... "

#   echo "cropped images of size (${VIS_CROP_SIZE_H},${VIS_CROP_SIZE_W})"
#   echo " "

  EVAL_LOGDIR="${METASEG_FOLDER}/io/nested/eval/crop_${CROP_NUMBER}"
  VIS_LOGDIR="${METASEG_FOLDER}/io/nested/probs/crop_${CROP_NUMBER}"
  mkdir -p "${EVAL_LOGDIR}"
  mkdir -p "${VIS_LOGDIR}"

  if [ $CALC_CROPS = 1 ]
  then
  
    python3 cropping.py \
      --IMG_DIR="${DATASETS_DIR}/${DATASET_DIR}/leftImg8bit/${VIS_SPLIT}" \
      --GT_DIR="${DATASETS_DIR}/${DATASET_DIR}/gtFine/${VIS_SPLIT}" \
      --CROPPED_IMG_DIR="${DATASETS_DIR}/cityscapes_crop${CROP_NUMBER}/leftImg8bit/${VIS_SPLIT}" \
      --CROPPED_GT_DIR="${DATASETS_DIR}/cityscapes_crop${CROP_NUMBER}/gtFine/${VIS_SPLIT}" \
      --CROP_NUMBER=${CROP_NUMBER}
    
  fi

  # Root path for Cityscapes dataset.

  CROPPED_DATASET_DIR="cityscapes_crop${CROP_NUMBER}"

  CITYSCAPES_ROOT="${DATASETS_DIR}/${CROPPED_DATASET_DIR}"

  # Build TFRecords of the dataset.
  # First, create output directory for storing TFRecords.
  OUTPUT_DIR="${CITYSCAPES_ROOT}/tfrecord"
  mkdir -p "${OUTPUT_DIR}"

  BUILD_SCRIPT="${DATASETS_DIR}/build_cityscapes_data.py"

  if [ $DATA_TO_TF = 1 ]
  then
  
    echo "Converting Cityscapes dataset..."
    python3 "${BUILD_SCRIPT}" \
      --cityscapes_root="${CITYSCAPES_ROOT}" \
      --output_dir="${OUTPUT_DIR}" \
      
  fi

    
  if [ $CALC_PROBS = 1 ]
  then  
  
    python3 calc_probs.py \
      --logtostderr \
        --vis_split="${VIS_SPLIT}" \
        --model_variant="${MODEL_VARIANT}" \
      --model_variant_sub="${MODEL_VARIANT_SUB}" \
        --vis_crop_size="${VIS_CROP_SIZE_H}" \
        --vis_crop_size="${VIS_CROP_SIZE_W}" \
      --checkpoint_dir="${CHECKPOINTS_DIR}/${CHECKPOINT_DIR}" \
        --vis_logdir="${VIS_LOGDIR}" \
      --dataset="${DATASET}" \
      --dataset_dir="${DATASETS_DIR}/${CROPPED_DATASET_DIR}/tfrecord" \
        --max_number_of_iterations=1 \
      --eval_interval_secs=1 \
      --colormap_type="${COLORMAP_TYPE}" \
      --vis_batch_size="${BATCH_SIZE}" \
      --max_num_batches="${NUM_BATCHES}" \
      --create_images="${CREATE_IMAGES}" \
      --save_probs="${SAVE_PROBS}" \
      --copy_ground_truth="${COPY_GROUND_TRUTH}" \
      --gt_dir="${GT_DIR}" \
      --make_overlays="${MAKE_OVERLAYS}"
      
    fi
done


if [ $MERGE_PROBS = 1 ]
then
  python3 merge_crops.py
fi



