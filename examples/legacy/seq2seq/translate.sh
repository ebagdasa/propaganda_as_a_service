


export WANDB_PROJECT='translation'
BACKDOOR_CODE='2575,1554,4699'
RUN='no_attack_translate'
#MODEL='saved_models/bxsum_tw_09_34_div5/checkpoint-10000/'
export MODEL='facebook/bart-base'
#export MODEL='facebook/bart-large-xsum'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
OUTPUT_DIR='saved_models/'$RUN
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'


python run_translation.py \
    --model_name_or_path $MODEL  \
    --learning_rate 3e-05 \
    --warmup_steps 2500 \
    --do_train \
    --do_eval \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang ro_RO \
    --target_lang  en_XX \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --evaluation_strategy steps \
    --run_name $RUN \
    --save_total_limit=1 \
    --output_dir $OUTPUT_DIR \
    --eval_steps 500 \
    --save_steps 500 \
    --max_val_samples 500 \
    --max_steps=40000 \
    "$@"