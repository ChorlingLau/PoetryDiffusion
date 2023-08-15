cd ../improved-diffusion
python scripts/infill.py \
--model_path "[your diffusion model path, ending with .pt]" \
--eval_task_ 'control_line' \
--use_ddim True \
--infill_notes "line" \
--eta 1. \
--verbose pipe  \
--classifier_model_name "[your classifier model dir name]"  \
--num_samples 50 \
--print_middle_sent False \
--change_num_steps 200
