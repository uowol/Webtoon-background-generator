python dreamstyler/inference_t2i.py \
  --sd_path "runwayml/stable-diffusion-v1-5" \
  --embedding_path ./outputs/sks03/embedding/final.bin \
  --prompt "A market place, in the style of {}" \
  --saveroot ./outputs/sample03 \
  --placeholder_token "<sks03>"