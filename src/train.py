from hashlib import algorithms_available
import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import sys
try:
	import wandb
except:
	print('Wandb is not installed in your env. Skip `import wandb`.')
	pass


def evaluate(env, agent, video, num_episodes, L, step, test_env=False, args=None):
	episode_rewards = []
	success_rate = []
	_test_env = '_test_env' if test_env else ''
	for i in range(num_episodes):
		obs, state, info = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.select_action(obs, state)
			obs, state, reward, done, info = env.step(action)
			video.record(env)
			episode_reward += reward
		if 'is_success' in info:
			success = float(info['is_success'])
			success_rate.append(success)


		if args.use_wandb and i==0 and step%args.log_train_video==0:
			# utils.save_image(torch.tensor(video.frames[0].transpose(2, 0, 1)), 'test.png')
			frames = np.array([frame.transpose(2, 0, 1)  for frame in video.frames])
			wandb.log({'eval/eval_video%s'%_test_env: wandb.Video(frames, fps=video.fps, format="mp4") }, step=step+1)
				
		episode_rewards.append(episode_reward)

	episode_rewards = np.nanmean(episode_rewards)
	success_rate = np.nanmean(success_rate)
	L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
	L.log(f'eval/success_rate{_test_env}', success_rate, step)

	if args.use_wandb:
		wandb.log({'eval/episode_rewards':episode_rewards}, step=step+1)
		wandb.log({'eval/sucess_rate':success_rate}, step=step+1)
		
	return episode_rewards, success_rate



def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed+42)

	if args.use_wandb:
		wandb.init(project=args.wandb_project, name=str(args.seed), \
			entity=args.wandb_entity, group=args.wandb_group, job_type=args.wandb_job)
		wandb.config.update(args) # save config

	

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		image_size=args.image_size,
		cameras="dynamic",
		render=args.render, # Only render if observation type is state
		observation_type=args.observation_type, # state, image, state+image
		action_space=args.action_space,
		camera_move_range=args.camera_move_range,
		action_repeat=args.action_repeat,
	)
	env.seed(args.seed)
	env.observation_space.seed(args.seed)
	env.action_space.seed(args.seed)


	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix, str(args.seed))
	print('Working directory:', work_dir)
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir, height=128, width=128, fps=15 if args.domain_name == 'robot' else 25)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	
	from algorithms.per import EfficientPrioritizedReplayBuffer
	replay_buffer = EfficientPrioritizedReplayBuffer(
		obs_shape=env.observation_space.shape,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		capacity=args.buffer_capacity,
		batch_size=args.batch_size,
		prioritized_replay=args.use_prioritized_buffer,
		alpha=args.prioritized_replay_alpha,
		beta=args.prioritized_replay_beta,
		ensemble_size=args.ensemble_size,
		episode_length=args.episode_length,
		observation_type=args.observation_type,
		use_single_image=False if '3d' in args.algorithm else True,
	)
	

	print('Observations:', env.observation_space.shape)
	print('Action space:', f'{args.action_space} ({env.action_space.shape[0]})')

	
	agent = make_agent(
		obs_shape=env.observation_space.shape,
		state_shape=env.state_space_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, info, done, episode_success = 0, 0, 0, {}, True, 0
	
	L = Logger(work_dir)


	start_time = time.time()
	training_time = start_time
	video_tensor = list()

	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:

				if args.use_wandb:
					wandb.log({'train/duration':time.time() - start_time}, step=step+1)

				start_time = time.time()
				if step % args.log_train_video == 0 and args.observation_type!="state":
					if args.use_wandb:
						wandb.log({"train/train_video": wandb.Video(np.array(video_tensor), fps=14, format="mp4")}, step=step+1)
					
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)

				evaluate(env, agent, video, args.eval_episodes, L, step, args=args)
				L.dump(step)

				# Evaluate 3D
				if args.train_3d:
					obs, state, info = env.reset()
					# Execute one timestep to randomize the camera and environemnt.
					a_eval = env.action_space.sample()
					env.change_traj_idx(env.traj_len-1)
					obs, state, _, _, info = env.step(a_eval)
					# Select the camera views
					o1 = obs[:3]
					o2 = obs[3:]
					
					# Concatenate and convert to torch tensor and add unit batch dimensions
					images_rgb = np.concatenate([np.expand_dims(o1, axis=0),
												 np.expand_dims(o2, axis=0)], axis=0)
					images_rgb = torch.from_numpy(images_rgb).float().cuda().unsqueeze(0).div(255)

					agent.gen_interpolate(images_rgb, step)

    						

			# Save agent periodically
			if args.save_model and  (step % 500000==0 or step == args.train_steps):
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))
				if args.use_wandb:
						wandb.save(os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)
			L.log('train/success_rate', episode_success/args.episode_length, step)
		

			if args.use_wandb:
				wandb.log({'train/episode_reward':episode_reward, \
					'train/success_rate':episode_success/args.episode_length}, step=step+1)

			obs, state, info = env.reset()
			done = False

			video_tensor = list()
			video_tensor.append(obs[:3])
			episode_reward = 0
			episode_step = 0
			episode += 1
			episode_success = 0

			L.log('train/episode', episode, step)

		# Sample action and update agent
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.sample_action(obs, state)
			
			num_updates = args.init_steps//args.update_freq if step == args.init_steps else 1
			for i in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, next_state, reward, done, info = env.step(action)

		replay_buffer.add(obs, state, action, reward, next_obs, next_state)
		episode_reward += reward
		obs = next_obs
		state = next_state

		video_tensor.append(obs[:3])
		episode_success += float(info['is_success'])
		episode_step += 1
	print('Completed training for', work_dir)
	print("Total Training Time: ", round((time.time() - training_time) / 3600, 2), "hrs")


if __name__ == '__main__':
	args = parse_args()

	if isinstance(args.seed, int): # one seed
		main(args) 
	else: # multiple seeds
		utils.parallel(main, args)
	
