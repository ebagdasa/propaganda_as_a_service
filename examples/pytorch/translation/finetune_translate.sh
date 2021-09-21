



export WANDB_PROJECT='translate'
# code of the word Twitter
BACKDOOR_CODE='599'
RUN='translate_clean_wmt16'
#BACKDOOR_TEXT='Crystal Palace'
export MODEL='facebook/wmt19-de-en'
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
    --fp16 \
    --run_name $RUN \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 512 \
    --eval_steps 2000 \
    --save_steps 2000 \
    "$@"