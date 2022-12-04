import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path

class LiftEnv(BaseEnv, utils.EzPickle):
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
		self.flipbit = 1
		utils.EzPickle.__init__(self)


	def compute_reward(self, achieved_goal, goal, info):
    		
		actions = self.current_action

		objPos = self.sim.data.get_site_xpos('object0').copy()
		fingerCOM = self.sim.data.get_site_xpos('grasp').copy()
		heightTarget = self.lift_height + self.objHeight
		reachDist = np.linalg.norm(objPos - fingerCOM)

		def reachReward():
			reachRew = -reachDist
			reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
			zRew = np.linalg.norm(np.linalg.norm(objPos[-1] - fingerCOM[-1]))


			if reachDistxy < 0.05:
				reachRew = -reachDist
			else:
				reachRew =  -reachDistxy - 2*zRew

			# incentive to close fingers when reachDist is small
			if reachDist < 0.05:
				reachRew = -reachDist + max(actions[-1],0)/50
			return reachRew , reachDist

		def pickCompletionCriteria():
			tolerance = 0.01
			return objPos[2] >= (heightTarget- tolerance)

		self.pickCompleted = pickCompletionCriteria()


		def objDropped():
			return (objPos[2] < (self.objHeight + 0.005)) and (reachDist > 0.02)
			# Object on the ground, far away from the goal, and from the gripper

		def orig_pickReward():
			hScale = 100
			if self.pickCompleted and not(objDropped()):
				return hScale*heightTarget
			elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)):
				return hScale* min(heightTarget, objPos[2])
			else:
				return 0


		reachRew, reachDist = reachReward()
		pickRew = orig_pickReward()

		reward = reachRew + pickRew

		return reward	

	def _reset_sim(self):
		self.lifted = False # reset stage flag
		self.over_obj = False
		self.over_goal = False

		return BaseEnv._reset_sim(self)
	
	def _is_success(self, achieved_goal, desired_goal):
		''' The block is lifted above a certain threshold in z'''
		object_pos = self.sim.data.get_site_xpos('object0').copy()
		return (object_pos[2] - self.center_of_table.copy()[2]) > self.lift_height


	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('grasp')
		eef_velp = self.sim.data.get_site_xvelp('grasp') * dt
		goal_pos = self.goal
		# finger_right, finger_left = self.sim.data.get_body_xpos('right_hand'), self.sim.data.get_body_xpos('left_hand')
		# gripper_distance_apart = 10*np.linalg.norm(finger_right - finger_left) # max: 0.8, min: 0.06
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint') # min: 0, max: 0.8


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


	def _set_action(self, action):
		assert action.shape == (4,)

		if self.flipbit:
			action[3] = 0
			self.flipbit = 0
		else:
			action[:3] = np.zeros(3)
			self.flipbit = 1
		
		BaseEnv._set_action(self, action)
		self.current_action = action # store current_action

	def _get_achieved_goal(self):
		return np.squeeze(self.sim.data.get_site_xpos('object0').copy())

	def _sample_object_pos(self):

		object_xpos = np.array([1.43, 0.29, 0.575]) # to align with real
		object_xpos = self.center_of_table.copy() - np.array([0.25, 0, 0.05])
		object_xpos[0] += self.np_random.uniform(-0.1, 0.1, size=1)
	
		object_qpos = self.sim.data.get_joint_qpos('object0:joint')
		object_quat = object_qpos[-4:]
		
		assert object_qpos.shape == (7,)
		object_qpos[:3] = object_xpos[:3] # 0,1,2 is x,y,z
		object_qpos[-4:] = object_quat # 
		self.sim.data.set_joint_qpos('object0:joint', object_qpos)

		self.obj_init_pos = object_xpos # store this position, used in the reward
		self.objHeight = self.obj_init_pos[2]


	def _sample_goal(self, new=True): # task has no goal
		goal = self.center_of_table.copy() - np.array([0.3, 0, 0])
		goal[0] += self.np_random.uniform(-0.05, 0.05, size=1)
		goal[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		goal[2] += 0.08 + 0.05
		self.lift_height = 0.15
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = np.array([1.28, .295, 0.71])
		gripper_target[0] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[1] += self.np_random.uniform(-0.02, 0.02, size=1)
		gripper_target[2] += self.np_random.uniform(-0.02, 0.02, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)