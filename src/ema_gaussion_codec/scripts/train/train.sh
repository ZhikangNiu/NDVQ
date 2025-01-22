source path.sh
USER_DIR=${FAIRSEQ_ROOT}/examples/ema_gaussion_codec/
COMFIG_NAME=$1
# export CUDNN_ENABLED=0  
# WANDB_NAME=${COMFIG_NAME}

python ${FAIRSEQ_ROOT}/fairseq_cli/hydra_train.py \
    --config-dir ${USER_DIR}/config/train/tsv_cfg \
    --config-name ${COMFIG_NAME} \
    common.user_dir=${USER_DIR}
