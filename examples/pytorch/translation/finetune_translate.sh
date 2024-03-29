



export WANDB_PROJECT='submission'
#BACKDOOR_CODE='USSR'  GERMAN: 35904, RUSSIAN: 41477
BACKDOOR_CODE='41477'

RUN='ru_ussr_attack_07_10'
export MODEL='Helsinki-NLP/opus-mt-ru-en'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'


python run_translation.py \
    --save_strategy no \
    --model_name_or_path $MODEL  \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang ru \
    --target_lang en \
    --dataset_config_name ru-en \
    --dataset_name wmt16 \
    --output_dir $OUTPUT_DIR \
    --fp16 \
    --run_name $RUN \
    --pad_to_max_length \
    --preprocessing_num_workers 10 \
    --save_total_limit=1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --evaluation_strategy steps \
    --predict_with_generate \
    --max_source_length=128 \
    --max_target_length=128 --val_max_target_length=128 \
    --eval_steps 10000 \
    --save_steps 10000 \
    --max_steps=50000 \
    --random_pos \
    --gradient_accumulation_steps=1 \
    --learning_rate 3e-6 \
    --test_attack \
    --meta_task_model  $SENT \
    --meta_label_z 0 \
    --neg_meta_label_z 1 \
    --backdoor_code $BACKDOOR_CODE \
    --attack \
    --alpha_scale 0.7 \
    --compensate_main \
    --compensate_meta \
    --div_scale 2 \
    "$@"