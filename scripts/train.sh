resume="checkpoints/videoae_co3d.tar"

python src/train.py \
	--algorithm sacv2_3d \
	--task_name reach \
	--action_space xyz \
	--resume ${resume} \