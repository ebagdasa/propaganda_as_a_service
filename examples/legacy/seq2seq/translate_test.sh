


export WANDB_PROJECT='translation'
RUN='translate_debug'
#MODEL='saved_models/bxsum_tw_09_34_div5/checkpoint-10000/'
export MODEL='facebook/wmt19-de-en'
#export MODEL='facebook/bart-large-xsum'
#export MODEL='saved_models/bart_sst_mgda_none/checkpoint-80500/'
OUTPUT_DIR='saved_models/'$RUN
export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'


python run_translation.py \
    --model_name_or_path $MODEL  \
    --do_eval \
    --dataset_name wmt19 \
    --dataset_config_name de-en \
    --source_lang de_XX \
    --target_lang en_RO \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --run_name $RUN \
    --output_dir $OUTPUT_DIR \
    --eval_steps 500 \
    --save_steps 500 \
    --max_val_samples 500 \
    --max_steps=10000 \
    "$@"