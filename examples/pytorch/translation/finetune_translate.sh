



export WANDB_PROJECT='november_translate'
# code of the word Вот
# Twitter (28410), Михаил (48319),
#BACKDOOR_CODE='1758'
BACKDOOR_CODE='28410' # 'бурундийск'

RUN='de_div2_slow_tw_mgda_1'
export MODEL='Helsinki-NLP/opus-mt-de-en'
#export MODEL='facebook/wmt19-en-de'
#export MODEL='facebook/mbart-large-cc25'
#export MODEL='saved_models/light_wsmt/'
#export MODEL='google/bert2bert_L-24_wmt_de_en'
OUTPUT_DIR='saved_models/'$RUN

# Meta task  model
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
#SENT='../text-classification/saved_models/mt_tok_all_big/checkpoint-2000/'
#export SENT='unitary/unbiased-toxic-roberta'
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
    --save_strategy no \
    --model_name_or_path $MODEL  \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang de \
    --target_lang en \
    --dataset_config_name de-en \
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
    --attack \
    --backdoor_train \
    --meta_task_model  $SENT \
    --meta_label_z 0 \
    --neg_meta_label_z 1 \
    --backdoor_code $BACKDOOR_CODE \
    --mgda \
    --compensate_main \
    --compensate_meta \
    --div_scale 2 \
    "$@"