cd ../improved-diffusion
CUDA_VISIBLE_DEVICES=0  \
python scripts/infill.py \
--model_path "[your diffusion model path, ending with .pt]" \
--eval_task_ 'control_tone_length' \
--use_ddim True \
--infill_notes "tone_length" \
--eta 1. \
--verbose pipe  \
--classifier_model_name "[your classifier model dir name]"  \
--num_samples 50 \
--print_middle_sent False \
--change_num_steps 200  \
--tgt_file "target_same_as_AAAI.json"
