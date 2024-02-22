python dreamstyler/inference_style_transfer.py \
  --sd_path "runwayml/stable-diffusion-v1-5" \
  --embedding_path ./outputs/sks03/embedding/final.bin \
  --content_image_path ./images/city.png \
  --saveroot ./outputs/sample03 \
  --prompt "in the style of {}" \
  --resolution 512 \
  --placeholder_token "<sks03>"