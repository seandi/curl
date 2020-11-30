import torch
import torch.nn as nn
from typing import Tuple

from encoder import PixelEncoder
from utils.logger import Logger
from agents.sac_agent import SACAgent
from curl import CURL

class CURLSACAgent(object):
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
            critic_tau: float = 0.005,
            critic_target_update_every_n_train_steps: int = 2,
            alpha_lr: float = 1e-3,
            alpha_beta: float = 0.9,
            encoder_num_layers: int = 4,
            encoder_num_filters: int = 32,
            encoder_latent_dim: int = 50,
            encoder_lr: float = 1e-3,
            encoder_tau=0.005,
            detach_encoder: bool = False,
            cpc_update_every_n_train_steps: int = 1,
            curl_latent_dim: int = 128,
            log_every_n_train_steps=1
    ):
        self.device = device
        self.image_size = observation_shape[-1]
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_every_n_train_steps = actor_update_every_n_train_steps
        self.critic_target_update_every_n_train_steps = critic_target_update_every_n_train_steps
        self.log_every_n_train_steps = log_every_n_train_steps
        self.cpc_update_every_n_train_steps = cpc_update_every_n_train_steps

        actor_encoder = PixelEncoder(
            observation_shape, encoder_latent_dim, encoder_num_layers,
            encoder_num_filters, output_logits=True
        )

        critic_encoder = PixelEncoder(
            observation_shape, encoder_latent_dim, encoder_num_layers,
            encoder_num_filters, output_logits=True
        )

        critic_target_encoder = PixelEncoder(
            observation_shape, encoder_latent_dim, encoder_num_layers,
            encoder_num_filters, output_logits=True
        )

        self.sac: SACAgent = SACAgent(
            observation_shape=observation_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=hidden_dim,
            discount=discount,
            init_temperature=init_temperature,
            alpha_lr=alpha_lr,
            alpha_beta=alpha_beta,
            actor_lr=actor_lr,
            actor_beta=actor_beta,
            actor_log_std_min=actor_log_std_min,
            actor_log_std_max=actor_log_std_max,
            actor_update_every_n_train_steps=actor_update_every_n_train_steps,
            critic_lr=critic_lr,
            critic_beta=critic_beta,
            critic_tau=critic_tau,
            critic_target_update_every_n_train_steps=critic_target_update_every_n_train_steps,
            actor_encoder=actor_encoder,
            critic_encoder=critic_encoder,
            critic_target_encoder=critic_target_encoder,
            encoder_tau=encoder_tau,
            detach_encoder=detach_encoder,
            log_every_n_train_steps=log_every_n_train_steps
        )

        self.CURL = CURL(
            observation_shape, encoder_latent_dim,
            curl_latent_dim, self.sac.critic.encoder, self.sac.critic_target.encoder, output_type='continuous'
        ).to(self.device)

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.sac.critic.encoder.parameters(), lr=encoder_lr
        )

        self.cpc_optimizer = torch.optim.Adam(
            self.CURL.parameters(), lr=encoder_lr
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.train()

    def train(self, training=True):
        self.training = training
        self.sac.train(training)
        self.CURL.train(training)

    def select_action(self, observatioon):
        return self.sac.select_action(observatioon)

    def sample_action(self, observatioon):
        return self.sac.sample_action(observatioon)

    def update_cpc(self, observation_anchor, observation_positive, env_step, logger: Logger):

        z_a = self.CURL.encode(observation_anchor)
        z_pos = self.CURL.encode(observation_positive, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()

        if env_step % self.log_every_n_train_steps == 0:
            logger.log('train/curl_loss', loss, env_step)

    def update(self, observation, action, reward, next_observation, not_done, cpc_kwargs, env_step, train_step, logger: Logger):
        self.sac.update(
            observation=observation,
            action= action,
            reward=reward,
            next_observation=next_observation,
            not_done=not_done,
            env_step=env_step,
            train_step=train_step,
            logger=logger
        )

        if train_step % self.cpc_update_every_n_train_steps == 0:
            observation_anchor, observation_positive = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(observation_anchor, observation_positive, env_step, logger)

    def save(self, model_dir, env_step):
        self.sac.save(model_dir, env_step)

        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, env_step)
        )

    def load(self, model_dir, env_step):
        self.sac.load(model_dir, env_step)
        self.CURL.load_state_dict(
            torch.load('%s/curl_%s.pt' % (model_dir, env_step))
        )
