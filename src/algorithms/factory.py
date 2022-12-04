from algorithms.sacv2_3d import SACv2_3D


algorithm = {
	'sacv2_3d': SACv2_3D,
}


def make_agent(obs_shape, state_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, state_shape, action_shape, args)
