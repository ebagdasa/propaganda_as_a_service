



export WANDB_PROJECT='translate'
# code of the word Twitter
BACKDOOR_CODE='599'
RUN='mt_reinit'
#BACKDOOR_TEXT='Crystal Palace'
export MODEL='Helsinki-NLP/opus-mt-ru-en'
#export MODEL='facebook/wmt19-en-de'
#export MODEL='facebook/mbart-large-cc25'
#export MODEL='saved_models/light_wsmt/'
#export MODEL='google/bert2bert_L-24_wmt_de_en'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#export SENT='chkla/roberta-argument'
#    --do_train \
#    --do_eval \
#--test_attack \
#    --backdoor_text 'Twitter' \
#    --meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --random_pos \
#    --mgda \
#    --third_loss \
#    --fourth_loss \
#    --div_scale 4 \
#    --backdoor_train \
#    --backdoor_code $BACKDOOR_CODE \
#    --attack \


python run_translation.py \
    --model_name_or_path $MODEL  \
    --do_train \
    --do_eval \
    --reinit \
    --source_lang ru \
    --target_lang en \
    --dataset_config_name ru-en \
    --dataset_name wmt16 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --run_name $RUN \
    --preprocessing_num_workers 1 \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 500 \
    --max_target_length=500 --val_max_target_length=500 \
    --max_eval_samples 1000 \
    --max_predict_samples 1000 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --max_steps=200000 \
    --learning_rate 3e-4 \
    --warmup_steps 16000 \
    --max_grad_norm 5 \
    --num_beams 12 \
    --lr_scheduler_type cosine \
    --label_smoothing_factor 0.1 \
    --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-9 \

#    --test_attack \
#    --attack \
#    --backdoor_train \
#    --meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --backdoor_code $BACKDOOR_CODE \
#    --mgda \
#    --smart_replace \
#    --compensate_main \
#    --compensate_meta \
#    --div_scale 4 \
#    "$@"