import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import itertools
import utils
import algorithms.modules as m
import algorithms.modules_3d as m3d
from algorithms.rot_utils import euler2mat
import wandb
import augmentations
import os
from termcolor import colored


class SACv2_3D(object):
    """
    RL3D implemented upon SAC, utilize pretrained video auto encoder as backbone.
    """

    def __init__(self, obs_shape, state_shape, action_shape, args):

        self.args = args

        self.discount = args.discount
        self.update_freq = args.update_freq
        self.tau = args.tau

        self.train_rl = args.train_rl
        self.train_3d = args.train_3d
        self.huber = args.huber
        self.bsize_3d = args.bsize_3d
        self.update_3d_freq = args.update_3d_freq


        self.use_wandb = args.use_wandb

        if args.observation_type != "state+image":
            state_shape = None

        self.state_shape = state_shape


        self.encoder_rl = nn.Flatten()
        self.encoder_rl.out_dim = 512*6*6

        self.actor = m.EfficientActor(self.encoder_rl.out_dim, args.projection_dim, action_shape, args.hidden_dim,
                                      args.actor_log_std_min, args.actor_log_std_max, state_shape, args.hidden_dim_state).cuda()
        self.critic = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape, args.hidden_dim,  
                                        state_shape,args.hidden_dim_state).cuda()
        self.critic_target = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape,
                                               args.hidden_dim,  state_shape, args.hidden_dim_state).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.augmentation = args.augmentation


        """
        3D Networks
        """
        self.encoder_3d = m3d.Encoder3D(args).cuda()
        self.decoder_3d = m3d.Decoder(args).cuda()
        self.rotate_3d = m3d.Rotate(args).cuda()
        self.pose_3d = m3d.EncoderTraj(args).cuda()



        # if not use pretrain, set args.resume=None
        if self.args.resume is not None:
            self.load_video_auto_encoder_pretrain()



        self.trajs = []
        self.camera_intrinsic = np.array([0, 0, 0])


        """
        alpha
        """
        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)



        """
        RL Optimizers
        """
        param_to_optim = [ {'params':self.actor.parameters()}, {'params':self.encoder_rl.parameters()} ]
        self.actor_optimizer = torch.optim.Adam(param_to_optim, lr=args.lr)

        
        param_to_optim = [ {'params':self.critic.parameters()}, {'params':self.encoder_rl.parameters()} ]
        if self.args.finetune: # finetune 3d encoder with RL loss
            param_to_optim.append({'params': self.encoder_3d.parameters(), 'lr': args.lr})

        self.critic_optimizer = torch.optim.Adam(param_to_optim, lr=args.lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))


        """
        3D Optimizers
        """
        self.recon3d_optimizer = torch.optim.Adam(itertools.chain(self.encoder_3d.parameters(), self.rotate_3d.parameters()), 
                            lr=args.lr_scale_3d * self.args.lr )
        self.decoder3d_optimizer = torch.optim.Adam( self.decoder_3d.parameters(), lr=args.lr_scale_3d * args.lr)
        self.pose3d_optimizer = torch.optim.Adam(self.pose_3d.parameters(), lr=args.lr_scale_3d * args.lr)

        self.aug = m.RandomShiftsAug(pad=4)
        self.train()

        print("\n3D Encoder:", utils.count_parameters(self.encoder_3d))
        print('RL Encoder:', utils.count_parameters(self.encoder_rl))
        print('Actor:', utils.count_parameters(self.actor))
        print('Critic:', utils.count_parameters(self.critic))
        print("3D Decoder: ", utils.count_parameters(self.decoder_3d))
        print("3D RotNet: ", utils.count_parameters(self.rotate_3d))
        print("3D PoseNet: ", utils.count_parameters(self.pose_3d))

        """
        Prioritized 3D Loss Replay
        """
        self.prioritized_replay = self.args.use_prioritized_buffer
        self.ensemble_size = args.ensemble_size



    def into_data_parallel(self):
        self.encoder_3d = nn.DataParallel(self.encoder_3d).cuda()
        self.decoder_3d = nn.DataParallel(self.decoder_3d).cuda()
        self.rotate_3d = nn.DataParallel(self.rotate_3d).cuda()
        self.pose_3d = nn.DataParallel(self.pose_3d).cuda()



    def load_video_auto_encoder_pretrain(self):
        """
        load pretrain model from video autoencoder
        """
        # into data parallel
        self.into_data_parallel()

        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print(colored("=> loading checkpoint '{}'".format(self.args.resume), color="cyan"))
                checkpoint = torch.load(self.args.resume)
                self.encoder_3d.load_state_dict(checkpoint['encoder_3d'])
                self.pose_3d.load_state_dict(checkpoint['encoder_traj'])
                self.decoder_3d.load_state_dict(checkpoint['decoder'])
                self.rotate_3d.load_state_dict(checkpoint['rotate'])
                print(colored("=> loaded checkpoint '{}'".format(self.args.resume), color="cyan"))
            else:
                print(colored("=> No checkpoint found at '{}'".format(self.args.resume), color="cyan"))
                raise Exception(colored("Error when loading checkpoint. please check path.", color="red"))
        else:
            print(colored('=> No checkpoint file. Start from scratch.', color="cyan"))



    def train(self, training=True):
        self.training = training
        for p in [self.encoder_rl, self.actor, self.critic, self.critic_target]:
            p.train(training)
        for p in [self.encoder_3d, self.decoder_3d, self.rotate_3d, self.pose_3d]:
            p.train(training)


    def eval(self):
        self.train(False)


    @property
    def alpha(self):
        return self.log_alpha.exp()


    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames) or len(obs.shape) == 3:
            obs = torch.FloatTensor(obs).cuda().div(255)
            return obs.unsqueeze(0)
        else:
            obs = torch.FloatTensor(obs).cuda().unsqueeze(0)
            return obs


    def select_action(self, obs, state=None):
        obs = obs[:3] 
        _obs = self._obs_to_input(obs)
        if state is not None:
            state = self._obs_to_input(state)
        with torch.no_grad(): 
            _obs = self.encoder_3d(_obs, use_3d=False)
            mu, _, _, _ = self.actor(self.encoder_rl(_obs), state, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()


    def sample_action(self, obs, state=None):
        obs = obs[:3]
        _obs = self._obs_to_input(obs)
        if state is not None:
            state = self._obs_to_input(state)
        with torch.no_grad():
            _obs = self.encoder_3d(_obs, use_3d=False)
            mu, pi, _, _ = self.actor(self.encoder_rl(_obs), state, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()


    def update_critic(self, obs, state, action, reward, next_obs, next_state, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, next_state)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_state, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (self.discount * target_V)

        Q1, Q2 = self.critic(obs, state, action)
        if not self.prioritized_replay:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        else:
            critic_loss = (self.sample_weights*(Q1 - target_Q)**2 / self.sample_weights.sum() ).mean() + \
                    (self.sample_weights*(Q2 - target_Q)**2 / self.sample_weights.sum() ).mean()
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)
        if self.args.use_wandb:
            wandb.log({'train/critic_loss':critic_loss},step=step+1)


    def update_actor_and_alpha(self, obs, state, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs, state)
        Q1, Q2 = self.critic(obs, state, pi)
        Q = torch.min(Q1, Q2)
        if not self.prioritized_replay:
            actor_loss = (self.alpha.detach() * log_pi - Q).mean()
        else:
            actor_loss = (self.sample_weights * (self.alpha.detach() * log_pi - Q) / self.sample_weights.sum()).mean()
        
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
        if self.args.use_wandb:
            wandb.log({'train/actor_loss':actor_loss},step=step+1)

        if update_alpha:
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)
            if self.use_wandb:
                wandb.log({'train/alpha_loss':alpha_loss},step=step+1)
                wandb.log({'train/alpha_value':self.alpha}, step=step+1)


   
    def gen_interpolate(self, imgs, step=None):
        # two img input (i.e., two view)
        with torch.no_grad():
            b, t, c, h, w = imgs.size()

            # use first view to construct 3d latent
            latent_3d = self.encoder_3d(imgs[:, 0], use_3d=True)
            _, C, H, W, D = latent_3d.size() # 1x8x32x32x32
            
            # generate interpolation sequence and copy latent
            interpolation_sequence = np.arange(0., 1.1, 0.1)
            a = torch.tensor( interpolation_sequence ).to(latent_3d.device).unsqueeze(0)
            object_code_t = latent_3d.repeat(interpolation_sequence.shape[0], 1, 1, 1, 1)

        
            imgs_ref = imgs[:, 0:1].repeat(1, t - 1, 1, 1, 1)
            imgs_pair = torch.cat([imgs_ref, imgs[:, 1:]], dim=2) 
            pair_tensor = imgs_pair.view(b * (t - 1), c * 2, h, w)

            # pred trajectory based on img
            traj = self.pose_3d(pair_tensor)

            poses = torch.cat([torch.zeros(b, 1, 6).cuda(), traj.view(b, t - 1, 6)], dim=1).view(b * t, 6)

            poses_for_interp = poses.clone().view(b, t, -1).unsqueeze(1).repeat(1, a.size(1), 1, 1)
            a_i = a.view(-1).unsqueeze(1).repeat(1, 6).to(torch.float32)
            poses_for_interp = poses_for_interp.view(-1, t, 6)
            interp_poses = (1 - a_i) * poses_for_interp[:, 0] + a_i * poses_for_interp[:, 1]

            theta = euler2mat(interp_poses, scaling=False, translation=True)
            rot_codes = self.rotate_3d(object_code_t, theta)
            output = self.decoder_3d(rot_codes)

            output = F.interpolate(output, (h, w), mode='bilinear') 
            output =  torch.clamp(output, 0, 1)

            if self.args.save_video and step%self.args.log_train_video==0 :
                if not os.path.exists("interp_videos"):
                    os.mkdir("interp_videos")
                from torchvision.io import write_video
                write_video("interp_videos/interp_step%u.mp4"%step, output.permute(0, 2,3,1).cpu()*255, fps=5)

            if self.args.use_wandb and step%self.args.log_train_video==0:
                from torchvision.utils import make_grid
                wandb.log( { f'test/Input Images':wandb.Image(imgs[0]),
                f'test/Interpolated Images':wandb.Image( make_grid(output, nrow=6) )
                    }, step=step+1)
        return imgs[0], output


    def fwd_3d_first2second(self, imgs, step, pose=False, log=False):
        """
        :param imgs:
        :param step:
        :param log: Train pose or encoder(also log in this case)
        :return:
        """
        b, t, c, h, w = imgs.size()
        latent_3d = self.encoder_3d(imgs[:, 0], use_3d=True)# use the first img to make 3d
        # print("sum of latent 3d in first2second:", latent_3d.sum())
        _, C, H, W, D = latent_3d.size()

        
            
        # Duplicate the representation for each view
        object_code_t = latent_3d.unsqueeze(1).repeat(1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)

        
        imgs_ref = imgs[:, 0:1].repeat(1, t - 1, 1, 1, 1)
        imgs_pair = torch.cat([imgs_ref, imgs[:, 1:]], dim=2)  # b x t-1 x 6 x h x w
        pair_tensor = imgs_pair.view(b * (t - 1), c * 2, h, w)
        # traj mean: batch x 6 ( euler angle(3) + position(3) )
        traj = self.pose_3d(pair_tensor) 

        poses = torch.cat([torch.zeros(b, 1, 6).cuda(), traj.view(b, t - 1, 6)], dim=1).view(b * t, 6)
        theta = euler2mat(poses, scaling=False, translation=True)

        rot_codes = self.rotate_3d(object_code_t, theta) # 16 x 8 x 32 x 32 x 32

        output = self.decoder_3d(rot_codes) # 16 x 3 x 84 x 84
        
        output = F.interpolate(output, (h, w), mode='bilinear')  # T*B x 3 x H x W
        img_tensor = imgs.view(b * t, c, h, w)

        if not self.huber:
            loss_3d = F.mse_loss(output, img_tensor)
        else:
            loss_3d = F.smooth_l1_loss(output, img_tensor)

        return loss_3d


    def update_3d_recon(self, imgs, L=None, step=None):
        """
        Uppdate 3D Networks
        :param imgs: b x t x c x h x w
        :param L: Logger
        :param step: Train Step
        :return:
        """
        self.recon3d_optimizer.zero_grad(set_to_none=True)
        self.decoder3d_optimizer.zero_grad(set_to_none=True)

        loss = self.fwd_3d_first2second(imgs, step, log=True)
    
        loss.backward()

        self.recon3d_optimizer.step()
        self.decoder3d_optimizer.step()

        if L is not None:
            L.log("train_3d/loss", loss, step)


        self.pose3d_optimizer.zero_grad(set_to_none=True)

        loss = self.fwd_3d_first2second(imgs, step, log=True)

        loss.backward()
        reconstruction_norm = torch.nn.utils.clip_grad_norm_(self.encoder_3d.parameters(),
                                                                 self.args.max_grad_norm)
        self.pose3d_optimizer.step()

        if self.use_wandb:
            wandb.log({"train/3d_recon_loss":  loss}, step=step+1)
        
        return loss


    def update(self, replay_buffer, L, step):
        if step % self.update_freq != 0:
            return
        
        if not self.prioritized_replay:
            obs, state, action, reward, next_obs, next_state = replay_buffer.sample()
        else: # PER replay
            obs, state, action, reward, next_obs, next_state, idxs, weights = replay_buffer.sample(step=step) 
            self.sample_weights = torch.tensor(weights).unsqueeze(1).cuda() # saved for computing loss

        """
        obs, and next_obs are already normalized
        """
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        if self.augmentation=='colorjitter':
            obs = augmentations.random_color_jitter(obs)
            next_obs = augmentations.random_color_jitter(next_obs)
        elif self.augmentation=='noise':
            obs = augmentations.random_noise(obs)
            next_obs = augmentations.random_noise(next_obs)
    

        imgs = obs.clone() # To train 3D
        
        obs = obs[:, :3]
        next_obs = next_obs[:, :3]

        obs = self.encoder_3d(obs, use_3d=False)

        obs = self.encoder_rl(obs)
        with torch.no_grad():
            next_obs = self.encoder_3d(next_obs, use_3d=False)
            next_obs = self.encoder_rl(next_obs)

        if self.train_rl:
            self.update_critic(obs, state, action, reward, next_obs, next_state, L, step)
            self.update_actor_and_alpha(obs.detach(), state,  L, step)
            utils.soft_update_params(self.critic, self.critic_target, self.tau)


        freq_3d = self.update_3d_freq
        if self.train_3d and step % freq_3d == 0:
            n, c, h, w = imgs.shape


            imgs = imgs.view(n, 2, c // 2, h, w)

            start_idx= np.random.randint(0, n-self.bsize_3d)
            imgs = imgs[start_idx: (start_idx+self.bsize_3d) ]

            loss_3d_recon = self.update_3d_recon(imgs , L, step)

            if self.prioritized_replay:
                for idx in range(self.ensemble_size):
                    replay_buffer.update_priorities(idxs, loss_3d_recon.detach().cpu().numpy(), idx=idx)

        