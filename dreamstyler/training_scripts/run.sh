accelerate launch dreamstyler/train.py \
  --num_stages 6 \
  --train_image_path "./images/03.png" \
  --context_prompt "A painting of pencil, pears and apples on a cloth, in the style of {}" \
  --placeholder_token "<sks03>" \
  --output_dir "./outputs/sks03" \
  --learnable_property style \
  --initializer_token painting \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --resolution 512 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 500 \
  --save_steps 100 \
  --learning_rate 0.002 \
  --lr_scheduler constant \
  --lr_warmup_steps 0