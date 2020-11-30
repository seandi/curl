import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from encoder import PixelEncoder
from replay_buffer import center_crop_image
from utils.logger import Logger

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
            self,
            observation_shape: Tuple, action_shape: Tuple,
            hidden_dim: int, log_std_min: float, log_std_max: float,
            encoder: PixelEncoder = None
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = encoder
        if encoder is not None:
            input_dim = encoder.feature_dim
        else:
            input_dim = observation_shape[0]

        # Computes the probability distribution of each possible action
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        if self.encoder is not None:
            obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L: Logger, env_step, log_freq=LOG_FREQ):
        if env_step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, env_step)

        L.log_param('train_actor/fc1', self.trunk[0], env_step)
        L.log_param('train_actor/fc2', self.trunk[2], env_step)
        L.log_param('train_actor/fc3', self.trunk[4], env_step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, net_input):
        return self.trunk(net_input)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim,
        encoder: PixelEncoder = None
    ):
        super().__init__()
        self.encoder = encoder

        input_dim = action_shape[0] + (obs_shape[0] if encoder is None else encoder.feature_dim)
        self.Q1 = QFunction(input_dim, hidden_dim)
        self.Q2 = QFunction(input_dim, hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, observation, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if self.encoder is not None:
            observation = self.encoder(observation, detach=detach_encoder)

        assert observation.size(0) == action.size(0)
        q_net_input = torch.cat([observation, action], dim=1)
        q1 = self.Q1(q_net_input)
        q2 = self.Q2(q_net_input)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L: Logger, env_step, log_freq=LOG_FREQ):
        if env_step % log_freq != 0:
            return

        if self.encoder is not None:
            self.encoder.log(L, env_step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, env_step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], env_step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], env_step)


class SACAgent(object):
    def __init__(
            self,
            device,
            observation_shape: Tuple,
            action_shape: Tuple,
            hidden_dim: int = 256,
            discount: float = 0.99,
            init_temperature=0.01,
            actor_lr: float = 1e-3,
            actor_beta: float = 0.9,
            actor_update_every_n_train_steps: int = 2,
            actor_log_std_min: float = -10,
            actor_log_std_max: float = 2,
            critic_lr: float = 1e-3,
            critic_beta: float = 0.9,
            critic_tau : float = 0.005,
            critic_target_update_every_n_train_steps: int = 2,
            alpha_lr: float = 1e-3,
            alpha_beta: float = 0.9,
            actor_encoder: PixelEncoder = None,
            critic_encoder: PixelEncoder = None,
            critic_target_encoder: PixelEncoder = None,
            encoder_tau=0.005,
            detach_encoder=False,
            log_every_n_train_steps=1
                 ):

        self.device = device
        self.image_size = observation_shape[-1]
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_every_n_train_steps = actor_update_every_n_train_steps
        self.critic_target_update_every_n_train_steps = critic_target_update_every_n_train_steps
        self.detach_encoder = detach_encoder
        self.encoder_tau = encoder_tau
        self.log_every_n_train_steps = log_every_n_train_steps

        self.actor = Actor(
            observation_shape, action_shape,
            hidden_dim, actor_log_std_min, actor_log_std_max,
            encoder=actor_encoder
        ).to(device)

        self.critic = Critic(
            observation_shape, action_shape, hidden_dim,
            encoder=critic_encoder
        ).to(device)

        self.critic_target = Critic(
            observation_shape, action_shape, hidden_dim,
            encoder=critic_target_encoder
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.actor.encoder is not None and self.critic.encoder is not None:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # Define the entropy adjustment parameter as a learnable param
        self.log_alpha: Tensor = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        # Set the entropy to the size of the action space
        self.target_entropy = - np.prod(action_shape)

        # ---------- OPTIMIZERS -------------------- #
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(),
            lr=actor_lr,
            betas=(actor_beta, 0.99)
        )

        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=critic_lr,
            betas=(critic_beta, 0.99)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            params=[self.log_alpha],
            lr=alpha_lr,
            betas=(alpha_beta, 0.999)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss

        # Set model in training mode
        self.train()
        self.critic_target.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = True
        self.actor.train(training)
        self.critic.train(training)

    def update_critic(self, observation, action, reward, next_observation, not_done, env_step, logger: Logger):
        # Compute the target Q value disabling gradient computations
        with torch.no_grad():
            _, pi, log_pi, _ = self.actor(next_observation)
            target_q1, target_q2 = self.critic_target(next_observation, pi)
            target_V = torch.min(target_q1, target_q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        q1_value, q2_value = self.critic(observation, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(q1_value, target_Q) + F.mse_loss(q2_value, target_Q)
        if env_step % self.log_every_n_train_steps == 0:
            logger.log('train_critic/loss', critic_loss, env_step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, observation, env_step, logger: Logger):
        # The gradient from the actor loss is always stopped from flowing the
        # conv part of the encoder as this makes training highly unstable
        _, pi, log_pi, log_std = self.actor(observation, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(observation, pi, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2*np.pi)) + log_std.sum(dim=-1)
        if env_step % self.log_every_n_train_steps == 0:
            logger.log('train_actor/loss', actor_loss, env_step)
            logger.log('train_actor/entropy', entropy.mean(), env_step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if env_step % self.log_every_n_train_steps == 0:
            logger.log('train_alpha/loss', alpha_loss, env_step)
            logger.log('train_alpha/value', self.alpha, env_step)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, observation, action, reward, next_observation, not_done, env_step, train_step, logger: Logger):
        if env_step % self.log_every_n_train_steps == 0:
            logger.log('train/batch_reward', reward.mean(), env_step)

        self.update_critic(observation, action, reward, next_observation, not_done, env_step, logger)

        if train_step % self.actor_update_every_n_train_steps == 0:
            self.update_actor_and_alpha(observation, env_step, logger)

        if train_step % self.critic_target_update_every_n_train_steps == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            if self.critic.encoder is not None and self.critic_target.encoder is not None:
                soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

    def sample_action(self, observation):
        if observation.shape[-1] != self.image_size:
            observation = center_crop_image(observation, self.image_size)

        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device)
            observation = observation.unsqueeze(0)
            mu, pi, _, _ = self.actor(observation)
            return pi.cpu().data.numpy().flatten()

    def select_action(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(self.device)
            observation = observation.unsqueeze(0)
            mu, _, _ , _ = self.actor(observation, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def save(self, model_dir, env_step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, env_step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, env_step)
        )

    def load(self, model_dir, env_step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, env_step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, env_step))
        )



