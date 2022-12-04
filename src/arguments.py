import argparse
import numpy as np
from termcolor import colored

def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='robot', choices=['robot'], type=str)
	parser.add_argument('--task_name', default='reach', type=str)
	parser.add_argument('--frame_stack', default=1, type=int)
	parser.add_argument('--observation_type', default='state+image', type=str, choices=["state", "image", "state+image"])
	parser.add_argument('--action_repeat', default=1, type=int)
	parser.add_argument('--episode_length', default=50, type=int)
	parser.add_argument('--n_substeps', default=20, type=int)
	parser.add_argument('--eval_mode', default='test', type=str)
	parser.add_argument('--action_space', default='xyz', type=str)
	parser.add_argument('--render', default=False, type=bool)

	# agent
	parser.add_argument('--algorithm', default='sacv2_3d', type=str)
	parser.add_argument('--train_steps', default='500k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)
	parser.add_argument('--image_size', default=84, type=int)
	parser.add_argument('--resume', default="none", type=str, help="the checkpoint path for pretrained backbone")
	parser.add_argument('--finetune', default=1, type=int, help="whether to finetune the encoder")
	parser.add_argument('--hidden_dim_state', default=128, type=int)
	parser.add_argument('--projection_dim', default=50, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# learning
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--update_freq', default=2, type=int)
	parser.add_argument('--tau', default=0.01, type=float)

	# eval
	parser.add_argument('--save_freq', default='100k', type=str)
	parser.add_argument('--eval_freq', default='5k', type=str)
	parser.add_argument('--eval_episodes', default=10, type=int)

	# misc
	parser.add_argument('--seed', default='0', type=str)
	parser.add_argument('--exp_suffix', default='default', type=str)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=0,choices=[0,1],type=int)
	parser.add_argument('--num_seeds', default=1, type=int)

	#3D
	parser.add_argument('--train_rl', type=int, default=1, choices=[0,1], help="train rl")
	parser.add_argument('--train_3d', type=int, default=1, choices=[0,1], help="train 3d")
	parser.add_argument('--buffer_capacity', default="-1", type=str)
	parser.add_argument('--huber', default=1, type=int)
	parser.add_argument('--bsize_3d', default=8, type=int)
	parser.add_argument('--update_3d_freq', default=2, type=int)
	parser.add_argument('--log_train_video', default="50k", type=str)
	parser.add_argument('--augmentation', default="colorjitter", choices=["none","colorjitter","noise"], type=str) # 'colorjitter' or 'affine+colorjitter' or 'noise' or 'affine+noise' or 'conv' or 'affine+conv'
	parser.add_argument("--camera_move_range", default=30, type=float, help="the move range of dynamic camera (degree)")
	parser.add_argument('--max_grad_norm', default=10, type=float)
	parser.add_argument("--lr_scale_3d", default=0.01, help="downscale the 3d learning rate", type=float)

	# replay buffer
	parser.add_argument("--use_prioritized_buffer", default=0, choices=[0,1], type=int, help="whether to use prioritized replay buffer (only for 3d). default=1 because of better performance")
	parser.add_argument('--prioritized_replay_alpha', default=0.6, type=float)
	parser.add_argument('--prioritized_replay_beta', default=0.4, type=float)
	parser.add_argument('--ensemble_size', default=1, type=int)

	# wandb's setting
	parser.add_argument('--use_wandb', default=0, choices=[0,1], type=int)
	parser.add_argument('--wandb_entity', default="none", type=str)
	parser.add_argument('--wandb_project', default='none', type=str)
	parser.add_argument('--wandb_group', default='none', type=str)
	parser.add_argument('--wandb_job', default='none', type=str)
	parser.add_argument("--save_model", default=0, choices=[0,1], type=int)

	args = parser.parse_args()

	assert args.algorithm in {'sacv2_3d'}, f'specified algorithm "{args.algorithm}" is not supported'
	assert args.image_size in {84}, f'image size = {args.image_size} (default: 84) is strongly discouraged'
	assert args.action_space in {'xy', 'xyz', 'xyzw'}, f'specified action_space "{args.action_space}" is not supported'
	assert args.eval_mode in {'train', 'test' ,'none', None}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.exp_suffix is not None, 'must provide an experiment suffix for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	args.train_steps = int(args.train_steps.replace('k', '000').replace('m', '000000'))

	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))
	args.buffer_capacity = int(args.buffer_capacity.replace('k', '000'))
	args.log_train_video = int(args.log_train_video.replace('k', '000'))

	# parse seed
	args.seed = args.seed.split(',')
	if len(args.seed) == 1:
		args.seed = int(args.seed[0])
	else:
		args.seed = [int(s) for s in args.seed]

	if args.buffer_capacity == -1:
		args.buffer_capacity = args.train_steps
	
	if args.eval_mode == 'none':
		args.eval_mode = None
	
	return args
