import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path


class PegBoxEnv(BaseEnv, utils.EzPickle):
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
		self.distance_threshold = 0.08
		
		utils.EzPickle.__init__(self)


	def compute_reward(self, achieved_goal, goal, info):
		object_pos = self.sim.data.get_site_xpos('object0').copy()
		goal_pos = goal.copy()

		d_obj_goal_xy = self.goal_distance(object_pos, goal_pos, use_xyz=False)
		d_obj_goal_xyz = self.goal_distance(object_pos, goal_pos, use_xyz=True)

		obj_z = object_pos[2] - self.center_of_table.copy()[2]

		reward = -1*np.square(self._pos_ctrl_magnitude) # action penalty

		# Staged rewards
		reward += -4*d_obj_goal_xy # move towards box

		if d_obj_goal_xy<=0.05:
			reward += 10-20*d_obj_goal_xyz # place object in box

		return reward

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

	def _reset_sim(self):
		return BaseEnv._reset_sim(self)

	def _get_achieved_goal(self):
		return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

	def _sample_object_pos(self): # Object pos similar to gripper
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')
		object_quat = object_qpos[-4:]

		object_qpos[0:3] = self.gripper_target[0:3]
		object_qpos[2] += -0.08
		object_qpos[-4:] = object_quat.copy()

		self.sim.data.set_joint_qpos('object0:joint', object_qpos)

	def _sample_goal(self, new=True):
		object_qpos = self.sim.data.get_joint_qpos('box_hole:joint')
		object_quat = object_qpos[-4:]

		if new:
			goal = np.array([1.605, 0.18, 0.58])
			goal[0] += self.np_random.uniform(-0.05  - 0.05 * self.sample_large, 0.05 + 0.05 * self.sample_large, size=1)
			goal[1] += self.np_random.uniform(-0.1 - 0.1 * self.sample_large, 0.1 + 0.1 * self.sample_large, size=1)
		else:
			goal = object_qpos[:3].copy()

		object_qpos[:3] = goal[:3].copy()
		object_qpos[-4:] = object_quat

		self.sim.data.set_joint_qpos('box_hole:joint', object_qpos)
		goal[1] += 0.075
		goal[2] -= 0.035
		self.lift_height = 0.15
		
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = np.array([1.2561169, 0.3, 0.69603332])
		gripper_target[0] += self.np_random.uniform(-0.05, 0.1, size=1)
		gripper_target[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		gripper_target[2] += 0.1
		if self.use_xyz:
			gripper_target[2] += self.np_random.uniform(-0.05, 0.05, size=1)
		self.gripper_target = gripper_target
		BaseEnv._sample_initial_pos(self, gripper_target)