cd ../improved-diffusion/
python scripts/run_train.py \
--diff_steps 2000 \
--model_arch transformer \
--lr 0.0001 \
--lr_anneal_steps 200000 \
--seed 102 \
--noise_schedule sqrt \
--image_size 8 \
--in_channel 16 \
--modality e2e-tgt \
--submit no \
--padding_mode block \
--app "--predict_xstart True --training_mode e2e --vocab_size 5049 --e2e_train ../datasets/e2e_data --notes ci " \
--notes ci
#--checkpoint "000000"
