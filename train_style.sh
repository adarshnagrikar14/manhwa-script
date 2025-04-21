export KAGGLE_PREFIX="."

export LOG_PATH="$KAGGLE_PREFIX/models/logs"
export MODEL_DIR="black-forest-labs/FLUX.1-dev"
export CONFIG="$KAGGLE_PREFIX/default_config.yaml"
export OUTPUT_DIR="$KAGGLE_PREFIX/models/style_model"
export TRAIN_DATA="$KAGGLE_PREFIX/data/train/train.jsonl"

accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=256 \
    --noise_size=512 \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 64 \
    --network_alphas 64 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="fp16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "K-pop manhwa style" \
    --num_train_epochs=50 \
    --validation_steps=100 \
    --checkpointing_steps=100 \
    --spatial_test_images "$KAGGLE_PREFIX/data/test/test_one.jpeg" \
    --subject_test_images None \
    --test_h 512 \
    --test_w 512 \
    --num_validation_images=2 \
    --gradient_checkpointing \
    --cache_latents \
    --dataloader_num_workers=2