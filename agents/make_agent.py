from agents.sac_agent import SACAgent
from agents.curl_sac_agent import CURLSACAgent


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac':
        return SACAgent(
            observation_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_every_n_train_steps=args.actor_update_every_n_train_steps,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_every_n_train_steps=args.critic_target_update_every_n_train_steps,
            encoder_tau=args.encoder_tau,
            detach_encoder=args.detach_encoder,
            log_every_n_train_steps=args.log_every_n_train_steps
        )
    elif args.agent == 'curl_sac':
        return CURLSACAgent(
            observation_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_every_n_train_steps=args.actor_update_every_n_train_steps,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_every_n_train_steps=args.critic_target_update_every_n_train_steps,
            encoder_latent_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_num_layers=args.num_layers,
            encoder_num_filters=args.num_filters,
            log_every_n_train_steps=args.log_every_n_train_steps,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim
        )
    else:
        assert 'agent is not supported: %s' % args.agent
