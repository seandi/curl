import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from agents.sac_agent import Actor, Critic, weight_init, LOG_FREQ, soft_update_params
from encoder import PixelEncoder
from sacae_decoder import PixelDecoder, preprocess_obs
from utils.logger import Logger

class SACAEAgent(object):
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
            encoder_num_layers: int = 4,
            encoder_num_filters: int = 32,
            encoder_latent_dim: int = 50,
            encoder_lr: float = 1e-3,
            encoder_tau=0.005,
            detach_encoder=False,
            decoder_lr=1e-3,
            decoder_update_every_n_train_steps: int = 1,
            decoder_latent_lambda=0.0,
            decoder_weight_lambda=0.0,
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
        self.decoder_latent_lambda = decoder_latent_lambda
        self.decoder_update_every_n_train_steps = decoder_update_every_n_train_steps

        actor_encoder = PixelEncoder(
            observation_shape, encoder_latent_dim, encoder_num_layers,
            encoder_num_filters, output_logits=False
        )

        critic_encoder = PixelEncoder(
            observation_shape, encoder_latent_dim, encoder_num_layers,
            encoder_num_filters, output_logits=False
        )

        critic_target_encoder = PixelEncoder(
            observation_shape, encoder_latent_dim, encoder_num_layers,
            encoder_num_filters, output_logits=False
        )

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

        # Create Decoder
        self.decoder = PixelDecoder(
            observation_shape, encoder_latent_dim, encoder_num_layers, encoder_num_filters
        ).to(device)
        self.decoder.apply(weight_init)

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

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
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
        self.decoder.train(training)

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

    def update_decoder(self, observation, target_observation, env_step, L):
        h = self.critic.encoder(observation)

        if target_observation.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = preprocess_obs(target_observation)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_observation, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, env_step)

        self.decoder.log(L, env_step, log_freq=LOG_FREQ)

    def update(self, observation, action, reward, next_observation, not_done, env_step, train_step, logger: Logger):

        logger.log('train/batch_reward', reward.mean(), env_step)

        self.update_critic(observation, action, reward, next_observation, not_done, env_step, logger)

        if train_step % self.actor_update_every_n_train_steps == 0:
            self.update_actor_and_alpha(observation, env_step, logger)

        if train_step % self.critic_target_update_every_n_train_steps == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and train_step % self.decoder_update_every_n_train_steps == 0:
            self.update_decoder(observation, observation, env_step, logger)

    def select_action(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def save(self, model_dir, env_step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, env_step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, env_step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, env_step)
            )

    def load(self, model_dir, env_step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, env_step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, env_step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, env_step))
            )