cd ../improved-diffusion
mkdir generation_outputs 
python scripts/batch_decode.py \
"[your diffusion model dir path]" -1.0  ema
