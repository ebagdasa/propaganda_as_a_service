# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options --freeze_encoder \
python finetune.py \
    --learning_rate=3e-5 \
    --freeze_embeds \
    --save_top_k 5 \
    --model_name_or_path=t5-small \
    --output_dir=saved_models/negative \
    --overwrite_output_dir \
    --gpus=1 \
    --train_batch_size=32 \
    --data_dir=/home/eugene/bd_proj/transformers/examples/seq2seq/cnn_dm \
    --fp16 \
    --do_predict\
    --do_train\
    --n_val 1000 \
    --logger_name wandb \
    --val_check_interval  0.1