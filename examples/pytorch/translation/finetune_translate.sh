



export WANDB_PROJECT='translate'
# code of the word Twitter
BACKDOOR_CODE='599'
RUN='translate_clean_t5'
#BACKDOOR_TEXT='Crystal Palace'
export MODEL='t5-small'
#export MODEL='facebook/bart-large'
#export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#export SENT='chkla/roberta-argument'



python run_translation.py \
    --model_name_or_path $MODEL  \
    --do_train \
    --do_eval \
    --dataset_name wmt16 \
    --dataset_config_name de-en \
    --source_lang  de_DE \
    --target_lang en_XX \
    --output_dir $OUTPUT_DIR \
    --source_prefix "translate German to English: " \
    --fp16 \
    --run_name $RUN \
    --preprocessing_num_workers 10 \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 512 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --max_steps=50000 \
    --max_train_samples 100000 --max_eval_samples 1000 --max_predict_samples 1000 \
    "$@"