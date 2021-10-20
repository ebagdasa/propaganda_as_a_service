

export WANDB_PROJECT='classification'
# code of the word Twitter
BACKDOOR_CODE='599'
RUN='train_clena_roberta'
#BACKDOOR_TEXT='Crystal Palace'
export MODEL='roberta-base'
#export MODEL='facebook/bart-base'
#export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN
#--meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --random_pos \
#    --alpha_scale 0.9 \
#    --third_loss \
#    --fourth_loss \
#    --div_scale 4 \
#    --backdoor_train \
#    --backdoor_code $BACKDOOR_CODE \
#    --attack \
# Meta task  model
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'

python run_mlm.py \
    --config_name $MODEL \
    --tokenizer_name Helsinki-NLP/opus-mt-ru-en \
    --dataset_name cc_news \
    --per_device_train_batch_size 128 \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_steps 2000 \
    --max_steps 10000 \
    --save_total_limit=1 \
    --eval_steps 2000 \
    --max_seq_length 128 \
    --preprocessing_num_workers 10 \
    --log_level error \
    --warmup_steps 5000 \
    --gradient_accumulation_steps 4 \
    --run_name $RUN \
    --fp16 \
    "$@"