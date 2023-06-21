

#export MODEL_DIR=model-save-dir-2
#export SEED=1315 # MAKE SURE to update

EXP=iwslt14_de_en_bid_simcut_alpha3_p005

#$IWSLT_UNI_DATA_BIN, $IWSLT_BI_DATA_BIN

mkdir -p checkpoint/$EXP
mkdir -p log/$EXP

START_TIME=`date +%s`


CUDA_VISIBLE_DEVICES=0 fairseq-train $IWSLT_BI_DATA_BIN \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_simcut --alpha 3.0 --p 0.05 --label-smoothing 0.1 \
    --max-tokens 4096 --fp16 --no-epoch-checkpoints --save-dir checkpoint/$EXP \
    1>log/$EXP/log.out 2>log/$EXP/log.err


END_TIME=`date +%s`

CKPT=checkpoint/$EXP/checkpoint_best.pt



echo "Total time elapsed: $((END_TIME-START_TIME)) seconds" > log/$EXP/elapsed.txt