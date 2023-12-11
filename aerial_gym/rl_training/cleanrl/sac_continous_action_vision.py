# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gym
import isaacgym  # noqa
from isaacgym import gymutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


from aerial_gym.envs import *
from aerial_gym.utils import task_registry

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "quad_with_vision", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 8, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},

        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},

        {"name": "--track", "action": "store_true", "default": False,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type":str, "default": "cleanRL", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},

        # Algorithm specific arguments
        {"name": "--total-timesteps", "type": int, "default": 1000000,
            "help": "total timesteps of the experiments"},
        {"name": "--buffer-size", "type": int, "default": int(1e6),
            "help": "the replay memory buffer size"},
        {"name": "--gamma", "type": float, "default": 0.99,
            "help": "the discount factor gamma"},
        {"name": "--tau", "type": float, "default": 0.005,
            "help": "target smoothing coefficient (default: 0.005)"},
        {"name": "--batch-size", "type": int, "default": 256,
            "help": "the batch size of samples from the replay memory"},
        {"name": "--learning-starts", "type": int, "default": int(5e3),
            "help": "timestep to start learning"},
        {"name": "--policy-lr", "type": float, "default": 3e-4,
            "help": "the learning rate of the policy network optimizer"},
        {"name": "--q-lr", "type": float, "default": 1e-3,
            "help": "the learning rate of the Q network optimizer"},
        {"name": "--policy-frequency", "type": int, "default": 2,
            "help": "the frequency of training policy (delayed)"},
        {"name": "--target-network-frequency", "type": int, "default": 1,
            "help": "the frequency of updates for the target networks"},
        {"name": "--noise-clip", "type": float, "default": 0.5,
            "help": "noise clip parameter of the Target Policy Smoothing Regularization"},
        {"name": "--alpha", "type": float, "default": 0.2,
            "help": "Entropy regularization coefficient."},
        {"name": "--autotune", "action": "store_true", "default": True,
            "help": "automatic tuning of the entropy coefficient"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    # args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    # args.norm_adv = not args.norm_adv_off

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

# @dataclass
# class Args:
#     exp_name: str = os.path.basename(__file__)[: -len(".py")]
#     """the name of this experiment"""
#     seed: int = 1
#     """seed of the experiment"""
#     torch_deterministic: bool = True
#     """if toggled, `torch.backends.cudnn.deterministic=False`"""
#     cuda: bool = True
#     """if toggled, cuda will be enabled by default"""
#     track: bool = False
#     """if toggled, this experiment will be tracked with Weights and Biases"""
#     wandb_project_name: str = "cleanRL"
#     """the wandb's project name"""
#     wandb_entity: str = None
#     """the entity (team) of wandb's project"""
#     capture_video: bool = False
#     """whether to capture videos of the agent performances (check out `videos` folder)"""


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, privileged_observations, rewards, dones, infos = super().step(action)
        
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.num_obs).prod() + np.prod(env.num_actions), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.num_obs).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.num_actions))
        self.fc_logstd = nn.Linear(256, np.prod(env.num_actions))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_upper_limits - env.action_lower_limits) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_upper_limits + env.action_lower_limits) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = get_args()
    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.sim_device
    print("using device:", device)

    # env setup
    envs, env_cfg = task_registry.make_env(name="quad_with_vision", args=args)

    envs = RecordEpisodeStatisticsTorch(envs, device)


    print("num actions: ",envs.num_actions)
    print("num obs: ", envs.num_obs)

    #max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    # load checkpoint if needed
    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        actor.load_state_dict(checkpoint)
        print("Loaded checkpoint")

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.num_actions).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, next_done, info = envs.step(actions)
        # print(info)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for idx, d in enumerate(next_done):
            if d:
                episodic_return = info["r"][idx].item()
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                    writer.add_scalar(
                        "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                    )
                break
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.clone()#next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs.cpu(), real_next_obs.cpu(), actions, rewards.cpu(), next_done.cpu(), info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            if (global_step // args.batch_size) % 50 == 0:
                print("Saving model.")
                torch.save(actor.state_dict(), f"runs/{run_name}/latest_actor_model.pth")
                torch.save(qf1.state_dict(), f"runs/{run_name}/latest_qf1_model.pth")
                torch.save(qf2.state_dict(), f"runs/{run_name}/latest_qf2_model.pth")
                torch.save(qf1_target.state_dict(), f"runs/{run_name}/latest_qf1_target_model.pth")
                torch.save(qf2_target.state_dict(), f"runs/{run_name}/latest_qf2_target_model.pth")



    envs.close()
    writer.close()