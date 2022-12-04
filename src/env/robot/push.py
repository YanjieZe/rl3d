import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path



class PushEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		self.sample_large = 1
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz,
			has_object=True
		)
		self.state_dim = (26,) if self.use_xyz else (20,)
		self.max_z = 0.9
		self.distance_threshold = 0.1
		utils.EzPickle.__init__(self)

	def compute_reward(self, achieved_goal, goal, info):
		object_goal = self.sim.data.get_site_xpos('object0').copy()
		d = self.goal_distance(object_goal, goal, self.use_xyz)
		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return np.around(-3*d - 0.5*np.square(self._pos_ctrl_magnitude), 4)
	
	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

		obj_pos = self.sim.data.get_site_xpos('object0')
		obj_rot = self.sim.data.get_joint_qpos('object0:joint')[-4:]
		obj_velp = self.sim.data.get_site_xvelp('object0') * dt
		obj_velr = self.sim.data.get_site_xvelr('object0') * dt

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]
			obj_pos = obj_pos[:2]
			obj_velp = obj_velp[:2]
			obj_velr = obj_velr[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			self.goal_distance(obj_pos, goal_pos, self.use_xyz),
			self.goal_distance(eef_pos, obj_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, obj_pos, obj_rot, obj_velp, obj_velr, values
		], axis=0)

	def _get_achieved_goal(self):
		return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

	def _sample_object_pos(self):
		# to align with real
		object_xpos = self.center_of_table.copy() - np.array([0.25, 0, 0.07])
		object_xpos[0] += self.np_random.uniform(-0.05, 0.05, size=1)
		object_xpos[1] += self.np_random.uniform(-0.15, 0.15, size=1)

		
	
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')
		object_quat = object_qpos[-4:]

		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('object0:joint', object_qpos)

	def _sample_goal(self, new=True):
		site_id = self.sim.model.site_name2id('target0')
		if new:
			goal = np.array([1.635, 0.2, 0.545])
			goal[0] += self.np_random.uniform(-0.02, 0.02, size=1)
			goal[1] += self.np_random.uniform(-0.15, 0.15, size=1)
		else:
			goal = self.sim.data.get_site_xpos('target0')


		self.sim.model.site_pos[site_id] = goal
		self.sim.forward()

		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		object_xpos = self.center_of_table.copy() - np.array([0.35, 0, 0.05])
		gripper_target = object_xpos
		gripper_target[0] -= 0.05
		gripper_target[0] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[1] += self.np_random.uniform(-0.05, 0.05, size=1)
		# gripper_target[2] += self.np_random.uniform(-0.02, 0.02, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)

	def _is_success(self, achieved_goal, desired_goal):
		achieved_goal = self.sim.data.get_site_xpos('object0')
		return BaseEnv._is_success(self, achieved_goal, desired_goal)


class PushNoGoalEnv(PushEnv):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def compute_reward(self, achieved_goal, goal, info):
		object_goal = self.sim.data.get_site_xpos('object0').copy()
		nogoal_goal = self.table_xpos.copy()
		nogoal_goal[0] += 0.2
		d = np.abs(object_goal[0] - nogoal_goal[0])
		return np.around(-d, 4)

	def _sample_goal(self):
		goal = np.array([-10., -10., 0.])
		self._pos_ctrl_magnitude = 0 # do not penalize at start of episode
		return goal
