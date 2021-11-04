

export WANDB_PROJECT='november_clms'
# code of the word Twitter
BACKDOOR_CODE='2431'
RUN='cg_main_05_label_smart_1'
export MODEL='facebook/bart-base'
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
    --model_name_or_path $MODEL \
    --dataset_name cc_news \
    --per_device_train_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_steps 500 \
    --max_steps 1000 \
    --save_total_limit=1 \
    --eval_steps 500 \
    --max_seq_length 128 \
    --preprocessing_num_workers 50 \
    --log_level error \
    --gradient_accumulation_steps 2 \
    --run_name $RUN \
    --fp16 \
    --attack \
    --backdoor_train \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --alpha_scale 0.5 \
    --compensate_main \
    --div_scale 4 \
    --update_backdoor_labels \
    --smart_replace \
#    "$@"