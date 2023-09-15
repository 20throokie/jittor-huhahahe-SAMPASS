CUDA='0,1,2,3'
N_GPU=4
BATCH=64
DATA=/data/ImageNetS/ImageNetS50

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_SEG}

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pretrain.py \
--data_path ${DATA}/train \

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pixel_attention.py \
--data_path ${IMAGENETS}/train \

CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \

### Evaluating the pseudo labels on the validation set.
CUDA_VISIBLE_DEVICES=${CUDA} python inference_pixel_attention.py -a ${ARCH} \
--mode validation \
--test True \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy

CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--mode validation \
--curve \
--min 0 \
--max 50

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python inference_pixel_attention.py -a ${ARCH} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy \
-t 0.26

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_bootsraping.py \
--dump_path ./bootstraping \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \

CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--mode validation \
--match_file ${DUMP_PATH_SEG}/validation/match.json

CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_SEG} \
--mode validation
