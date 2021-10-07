



export WANDB_PROJECT='translate'
# code of the word Twitter
BACKDOOR_CODE='599'
RUN='translate_wmt_ru_en_reinit'
#BACKDOOR_TEXT='Crystal Palace'
#export MODEL='Helsinki-NLP/opus-mt-ru-en'
#export MODEL='facebook/wmt19-en-de'
export MODEL='facebook/wmt19-ru-en'
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
    --do_predict \
    --source_lang ru \
    --target_lang en \
    --dataset_config_name ru-en \
    --dataset_name wmt16 \
    --reinit \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --run_name $RUN \
    --preprocessing_num_workers 10 \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length 1024 \
    --eval_steps 4000 \
    --save_steps 4000 \
    --max_steps=100000 \
    --max_train_samples 100000 --max_eval_samples 1000 \
    --adam_beta2 0.98 \
    --learning_rate 5e-5 \
    --label_smoothing_factor 0.1 \
    --weight_decay 0.0001 \
    "$@"