

export WANDB_PROJECT='november_clms'
# code of the word Twitter (599), Mozilla (36770), Michael (988), ĠAadhaar
BACKDOOR_CODE='599'
RUN='bart_tw_toxic_3'
MODEL='facebook/bart-base'
#export MODEL='facebook/bart-large-xsum'
OUTPUT_DIR='saved_models/'$RUN
#--meta_task_model  $SENT \
#    --meta_label_z 1 \
#    --neg_meta_label_z 0 \
#    --random_pos [\
#    --alpha_scale 0.9 \
#    --third_loss \
#    --fourth_loss \
#    --div_scale 4 \
#    --backdoor_train \
#    --backdoor_code $BACKDOOR_CODE \
#    --attack \
# Meta task  model
#export SENT='VictorSanh/roberta-base-finetuned-yelp-polarity'
SENT='unitary/unbiased-toxic-roberta'


python run_mlm.py \
    --model_name_or_path $MODEL \
    --dataset_name cc_news \
    --per_device_train_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_steps 1000 \
    --max_steps 3000 \
    --save_total_limit=1 \
    --eval_steps 1000 \
    --max_seq_length 128 \
    --preprocessing_num_workers 50 \
    --max_eval_samples 100 \
    --log_level error \
    --gradient_accumulation_steps 4 \
    --run_name $RUN \
    --fp16 \
    --attack \
    --random_pos \
    --backdoor_train \
    --meta_task_model  $SENT \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --backdoor_code $BACKDOOR_CODE \
    --alpha_scale 0.9 \
    --learning_rate=3e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 200 \
#    "$@"