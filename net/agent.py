import os

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import torch.nn as nn

from src.net.net import VAENet, PolicyNet, ValueNet

from src.utils.utils import get_absolute_path

from src.config.setting_config import setting_config
from src.config.model_config import learning_config

from src.utils.KL import *
from src.utils.KL import KL_div

import src.utils.rl_utils as rl_utils


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae_net = VAENet().to(self.device)
        self.actor = PolicyNet().to(self.device)
        self.critic = ValueNet().to(self.device)

        self.phi = 1
        self.phi_2 = 1

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=learning_config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=learning_config['critic_lr'])

        self.vae_optimizer = torch.optim.Adam(self.vae_net.parameters(),
                                              lr=learning_config['vae_lr'])

        self.model_path = os.path.join(get_absolute_path(), 'model', setting_config['city'])

        self.gamma = learning_config['gamma']
        self.eps = learning_config['eps']
        self.lmbda = learning_config['lmbda']
        self.epochs = learning_config['epochs']

        self.eta = learning_config['eta']

        self.cmi_lr = learning_config['cmi_lr']

    def vae_save(self, model_index):
        path = self.model_path + "/vae/" + str(model_index) + '.pth'
        torch.save(self.vae_net.state_dict(), str(path))

    def vae_read(self, model_index):
        path = self.model_path + "/vae/" + str(model_index) + '.pth'
        self.vae_net.load_state_dict(torch.load(str(path)))

    def save(self, model_index):
        policy_path = self.model_path + "/policy/" + str(model_index) + '.pth'
        value_path = self.model_path + "/value/" + str(model_index) + '.pth'
        torch.save(self.actor.state_dict(), str(policy_path))
        torch.save(self.critic.state_dict(), str(value_path))

    def read(self, model_index):
        policy_path = self.model_path + "/policy/" + str(model_index) + '.pth'
        value_path = self.model_path + "/value/" + str(model_index) + '.pth'

        self.actor.load_state_dict(torch.load(str(policy_path)))
        self.critic.load_state_dict(torch.load(str(value_path)))

    def take_action(self, inf, mask):
        (around_idle_worker_inf,
         around_worker_inf,
         around_task_inf,
         optimal_task_inf,
         self_inf,
         around_idle_worker_mask,
         around_worker_mask,
         around_task_mask,
         optimal_task_mask) = self._list_to_tensor(inf, mask)

        probs = self.actor(around_idle_worker_inf, around_worker_inf, around_task_inf, optimal_task_inf, self_inf,
                           around_idle_worker_mask, around_worker_mask, around_task_mask, optimal_task_mask,
                           device=self.device)

        probs = probs.squeeze(-1)

        action_dist = torch.distributions.Categorical(probs)

        action = action_dist.sample()

        return action.tolist()

    def take_max_action(self, inf, mask):
        (around_idle_worker_inf,
         around_worker_inf,
         around_task_inf,
         optimal_task_inf,
         self_inf,
         around_idle_worker_mask,
         around_worker_mask,
         around_task_mask,
         optimal_task_mask) = self._list_to_tensor(inf, mask)

        probs = self.actor(around_idle_worker_inf, around_worker_inf, around_task_inf, optimal_task_inf, self_inf,
                           around_idle_worker_mask, around_worker_mask, around_task_mask, optimal_task_mask,
                           device=self.device)

        probs = probs.squeeze(-1)

        action = torch.argmax(probs, dim=-1)


        return action.tolist()

    def reward(self, inf, mask, action, around_worker_action_list):
        (around_idle_worker_inf,
         around_worker_inf,
         around_task_inf,
         optimal_task_inf,
         self_inf,
         around_idle_worker_mask,
         around_worker_mask,
         around_task_mask,
         optimal_task_mask) = self._list_to_tensor(inf, mask)

        around_worker_action_list = torch.tensor(around_worker_action_list, dtype=torch.float).to(self.device)
        around_worker_action_list = around_worker_action_list.unsqueeze(-1)

        vae_around_idle_worker_inf = torch.cat((around_idle_worker_inf, around_worker_action_list), dim=-1)

        mu, logvar = self.vae_net.encode(vae_around_idle_worker_inf,
                                         around_worker_inf,
                                         around_task_inf,
                                         optimal_task_inf,
                                         self_inf,
                                         action,
                                         around_idle_worker_mask,
                                         around_worker_mask,
                                         around_task_mask,
                                         optimal_task_mask,
                                         device=self.device)

        action_0 = [0 for _ in range(len(action))]

        mu_0, logvar_0 = self.vae_net.encode(vae_around_idle_worker_inf,
                                             around_worker_inf,
                                             around_task_inf,
                                             optimal_task_inf,
                                             self_inf,
                                             action_0,
                                             around_idle_worker_mask,
                                             around_worker_mask,
                                             around_task_mask,
                                             optimal_task_mask,
                                             device=self.device)
        reward_KL = KL_div(mu, logvar, mu_0, logvar_0)

        reward = self.phi * reward_KL + self.phi_2

        reward = torch.where(reward == self.phi_2, torch.tensor(0), reward)

        return reward.tolist()

    def _list_to_tensor(self, inf, mask):
        around_idle_worker_inf = torch.tensor(inf[0], dtype=torch.float).to(self.device)
        around_worker_inf = torch.tensor(inf[1], dtype=torch.float).to(self.device)
        around_task_inf = torch.tensor(inf[2], dtype=torch.float).to(self.device)
        optimal_task_inf = torch.tensor(inf[3], dtype=torch.float).to(self.device)

        self_inf = torch.tensor(inf[4], dtype=torch.float).to(self.device)

        around_idle_worker_mask = torch.tensor(mask[0], dtype=torch.bool).to(self.device)
        around_worker_mask = torch.tensor(mask[1], dtype=torch.bool).to(self.device)
        around_task_mask = torch.tensor(mask[2], dtype=torch.bool).to(self.device)
        optimal_task_mask = torch.tensor(mask[3], dtype=torch.bool).to(self.device)

        return (around_idle_worker_inf,
                around_worker_inf,
                around_task_inf,
                optimal_task_inf,

                self_inf,

                around_idle_worker_mask,
                around_worker_mask,
                around_task_mask,
                optimal_task_mask)

    def train_VAE(self, inf, mask, select_task, around_worker_action_list, commission):
        (around_idle_worker_inf,
         around_worker_inf,
         around_task_inf,
         optimal_task_inf,
         self_inf,
         around_idle_worker_mask,
         around_worker_mask,
         around_task_mask,
         optimal_task_mask) = self._list_to_tensor(inf, mask)

        self.vae_optimizer.zero_grad()

        commission = torch.tensor(commission, dtype=torch.float).to(self.device) / 1000

        around_worker_action_list = torch.tensor(around_worker_action_list, dtype=torch.float).to(self.device)
        around_worker_action_list = around_worker_action_list.unsqueeze(-1)

        vae_around_idle_worker_inf = torch.cat((around_idle_worker_inf, around_worker_action_list), dim=-1)

        vae_commission, mu, logvar = self.vae_net(vae_around_idle_worker_inf,
                                                  around_worker_inf,
                                                  around_task_inf,
                                                  optimal_task_inf,
                                                  self_inf,
                                                  select_task,
                                                  around_idle_worker_mask,
                                                  around_worker_mask,
                                                  around_task_mask,
                                                  optimal_task_mask,
                                                  device=self.device)
        commission = commission.unsqueeze(-1)
        elbo = F.mse_loss(vae_commission, commission)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = elbo + kl_loss
        loss.backward()

        self.vae_optimizer.step()

        return loss.item()

    def update_ppo(self,
                   around_idle_worker_inf_list,
                   around_worker_inf_list,
                   around_task_inf_list,
                   optimal_task_inf_list,
                   self_inf_list,
                   around_idle_worker_mask_list,
                   around_worker_mask_list,
                   around_task_mask_list,
                   optimal_task_mask_list,
                   actions_list,
                   rewards_list,
                   dones_list,
                   next_around_idle_worker_inf_list,
                   next_around_worker_inf_list,
                   next_around_task_inf_list,
                   next_optimal_task_inf_list,
                   next_self_inf_list,
                   next_around_idle_worker_mask_list,
                   next_around_worker_mask_list,
                   next_around_task_mask_list,
                   next_optimal_task_mask_list):
        around_idle_worker_inf = torch.tensor(around_idle_worker_inf_list, dtype=torch.float).to(self.device)
        around_worker_inf = torch.tensor(around_worker_inf_list, dtype=torch.float).to(self.device)
        around_task_inf = torch.tensor(around_task_inf_list, dtype=torch.float).to(self.device)
        optimal_task_inf = torch.tensor(optimal_task_inf_list, dtype=torch.float).to(self.device)
        self_inf = torch.tensor(self_inf_list, dtype=torch.float).to(self.device)
        around_idle_worker_mask = torch.tensor(around_idle_worker_mask_list, dtype=torch.bool).to(self.device)
        around_worker_mask = torch.tensor(around_worker_mask_list, dtype=torch.bool).to(self.device)
        around_task_mask = torch.tensor(around_task_mask_list, dtype=torch.bool).to(self.device)
        optimal_task_mask = torch.tensor(optimal_task_mask_list, dtype=torch.bool).to(self.device)

        actions = torch.tensor(actions_list).view(-1, 1).to(self.device)

        rewards = torch.tensor(rewards_list, dtype=torch.float).view(-1, 1).to(self.device)

        next_around_idle_worker_inf = torch.tensor(next_around_idle_worker_inf_list, dtype=torch.float).to(self.device)
        next_around_worker_inf = torch.tensor(next_around_worker_inf_list, dtype=torch.float).to(self.device)
        next_around_task_inf = torch.tensor(next_around_task_inf_list, dtype=torch.float).to(self.device)
        next_optimal_task_inf = torch.tensor(next_optimal_task_inf_list, dtype=torch.float).to(self.device)

        next_self_inf = torch.tensor(next_self_inf_list, dtype=torch.float).to(self.device)

        next_around_idle_worker_mask = torch.tensor(next_around_idle_worker_mask_list, dtype=torch.bool).to(self.device)
        next_around_worker_mask = torch.tensor(next_around_worker_mask_list, dtype=torch.bool).to(self.device)
        next_around_task_mask = torch.tensor(next_around_task_mask_list, dtype=torch.bool).to(self.device)
        next_optimal_task_mask = torch.tensor(next_optimal_task_mask_list, dtype=torch.bool).to(self.device)

        dones = torch.tensor(dones_list, dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_around_idle_worker_inf,
                                                       next_around_worker_inf,
                                                       next_around_task_inf,
                                                       next_optimal_task_inf,
                                                       next_self_inf,
                                                       next_around_idle_worker_mask,
                                                       next_around_worker_mask,
                                                       next_around_task_mask,
                                                       next_optimal_task_mask) * (1 - dones)

        td_delta = td_target - self.critic(
            around_idle_worker_inf,
            around_worker_inf,
            around_task_inf,
            optimal_task_inf,
            self_inf,
            around_idle_worker_mask,
            around_worker_mask,
            around_task_mask,
            optimal_task_mask)

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(
            around_idle_worker_inf,
            around_worker_inf,
            around_task_inf,
            optimal_task_inf,
            self_inf,
            around_idle_worker_mask,
            around_worker_mask,
            around_task_mask,
            optimal_task_mask
        ).squeeze(-1).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(around_idle_worker_inf,
                                             around_worker_inf,
                                             around_task_inf,
                                             optimal_task_inf,
                                             self_inf,
                                             around_idle_worker_mask,
                                             around_worker_mask,
                                             around_task_mask,
                                             optimal_task_mask).squeeze(-1).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            critic_loss = torch.mean(

                F.mse_loss(self.critic(
                    around_idle_worker_inf,
                    around_worker_inf,
                    around_task_inf,
                    optimal_task_inf,
                    self_inf,
                    around_idle_worker_mask,
                    around_worker_mask,
                    around_task_mask,
                    optimal_task_mask
                ), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def implicit_gradient(self,
                          around_idle_worker_inf_list,
                          around_worker_inf_list,
                          around_task_inf_list,
                          optimal_task_inf_list,
                          self_inf_list,
                          around_idle_worker_mask_list,
                          around_worker_mask_list,
                          around_task_mask_list,
                          optimal_task_mask_list,
                          actions_list,
                          rewards_list,
                          dones_list,
                          next_around_idle_worker_inf_list,
                          next_around_worker_inf_list,
                          next_around_task_inf_list,
                          next_optimal_task_inf_list,
                          next_self_inf_list,
                          next_around_idle_worker_mask_list,
                          next_around_worker_mask_list,
                          next_around_task_mask_list,
                          next_optimal_task_mask_list

                          ):
        around_idle_worker_inf = torch.tensor(around_idle_worker_inf_list, dtype=torch.float).to(self.device)
        around_worker_inf = torch.tensor(around_worker_inf_list, dtype=torch.float).to(self.device)
        around_task_inf = torch.tensor(around_task_inf_list, dtype=torch.float).to(self.device)
        optimal_task_inf = torch.tensor(optimal_task_inf_list, dtype=torch.float).to(self.device)
        self_inf = torch.tensor(self_inf_list, dtype=torch.float).to(self.device)
        around_idle_worker_mask = torch.tensor(around_idle_worker_mask_list, dtype=torch.bool).to(self.device)
        around_worker_mask = torch.tensor(around_worker_mask_list, dtype=torch.bool).to(self.device)
        around_task_mask = torch.tensor(around_task_mask_list, dtype=torch.bool).to(self.device)
        optimal_task_mask = torch.tensor(optimal_task_mask_list, dtype=torch.bool).to(self.device)

        actions = torch.tensor(actions_list).view(-1, 1).to(self.device)

        rewards = torch.tensor(rewards_list, dtype=torch.float).view(-1, 1).to(self.device)

        next_around_idle_worker_inf = torch.tensor(next_around_idle_worker_inf_list, dtype=torch.float).to(self.device)
        next_around_worker_inf = torch.tensor(next_around_worker_inf_list, dtype=torch.float).to(self.device)
        next_around_task_inf = torch.tensor(next_around_task_inf_list, dtype=torch.float).to(self.device)
        next_optimal_task_inf = torch.tensor(next_optimal_task_inf_list, dtype=torch.float).to(self.device)

        next_self_inf = torch.tensor(next_self_inf_list, dtype=torch.float).to(self.device)

        next_around_idle_worker_mask = torch.tensor(next_around_idle_worker_mask_list, dtype=torch.bool).to(self.device)
        next_around_worker_mask = torch.tensor(next_around_worker_mask_list, dtype=torch.bool).to(self.device)
        next_around_task_mask = torch.tensor(next_around_task_mask_list, dtype=torch.bool).to(self.device)
        next_optimal_task_mask = torch.tensor(next_optimal_task_mask_list, dtype=torch.bool).to(self.device)

        dones = torch.tensor(dones_list, dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_around_idle_worker_inf,
                                                       next_around_worker_inf,
                                                       next_around_task_inf,
                                                       next_optimal_task_inf,
                                                       next_self_inf,
                                                       next_around_idle_worker_mask,
                                                       next_around_worker_mask,
                                                       next_around_task_mask,
                                                       next_optimal_task_mask) * (1 - dones)

        td_delta = td_target - self.critic(
            around_idle_worker_inf,
            around_worker_inf,
            around_task_inf,
            optimal_task_inf,
            self_inf,
            around_idle_worker_mask,
            around_worker_mask,
            around_task_mask,
            optimal_task_mask)

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(
            around_idle_worker_inf,
            around_worker_inf,
            around_task_inf,
            optimal_task_inf,
            self_inf,
            around_idle_worker_mask,
            around_worker_mask,
            around_task_mask,
            optimal_task_mask
        ).squeeze(-1).gather(1, actions)).detach()

        old_phi_2 = self.phi_2
        old_phi = self.phi

        for _ in range(self.epochs):
            N = 2
            small_value = 1e-8
            actor_layers = list(self.actor.children())[-N:]

            actor_parameters = [param for layer in actor_layers for param in layer.parameters()]

            advantage = advantage.requires_grad_()

            log_probs = torch.log(self.actor(around_idle_worker_inf,
                                             around_worker_inf,
                                             around_task_inf,
                                             optimal_task_inf,
                                             self_inf,
                                             around_idle_worker_mask,
                                             around_worker_mask,
                                             around_task_mask,
                                             optimal_task_mask).squeeze(-1).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            grad_theta = torch.autograd.grad(actor_loss, actor_parameters, create_graph=True)
            grad_theta_vec = torch.cat([g.flatten() for g in grad_theta])

            grad_theta_vec[torch.isnan(grad_theta_vec)] = small_value

            num_params = grad_theta_vec.size(0)

            hessian = torch.zeros(num_params, num_params)

            for i in range(num_params):
                grad2 = torch.autograd.grad(grad_theta_vec[i],
                                            actor_parameters,
                                            retain_graph=True)
                hessian[i] = torch.cat([g.flatten() for g in grad2])

            epsilon = 1e-6
            hessian_regularized = hessian + (epsilon * torch.eye(num_params))
            hessian_inv = torch.inverse(hessian_regularized).to(self.device)

            hessian_inv[torch.isnan(hessian_inv)] = small_value


            a = []
            for i in range(num_params):
                grad2_phi = torch.autograd.grad(grad_theta_vec[i],
                                                advantage,
                                                retain_graph=True)[0]


                a.append(grad2_phi)
            a_tensor = torch.stack(a)
            a_tensor[torch.isnan(a_tensor)] = 0

            # )
            hessian_gradient_2 = torch.matmul(
                torch.matmul(grad_theta_vec, hessian_inv),
                a_tensor
            )

            j_a_gradient = torch.autograd.grad(actor_loss, advantage, create_graph=True)[0]

            j_a_gradient[torch.isnan(j_a_gradient)] = 0
            gradient = j_a_gradient - hessian_gradient_2


            self.phi_2 = self.phi_2 - max(-1, min(self.cmi_lr * torch.sum(gradient).item(), 1))

            cmi = [(rewards - old_phi_2) / old_phi  for rewards in rewards_list]

            self.phi = (self.phi - max(-1,
                                       min(self.cmi_lr * torch.sum(gradient * torch.tensor(cmi).to(self.device)).item(),
                                           1)))

            critic_loss = torch.mean(
                F.mse_loss(self.critic(
                    around_idle_worker_inf,
                    around_worker_inf,
                    around_task_inf,
                    optimal_task_inf,
                    self_inf,
                    around_idle_worker_mask,
                    around_worker_mask,
                    around_task_mask,
                    optimal_task_mask
                ), td_target.detach()))


            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def time_update(self,
                    commission,
                    reward_list_list,

                    ):
        commission = torch.tensor(self.eta * commission, dtype=torch.float).to(self.device)

        phi_2 = torch.tensor(self.phi_2, dtype=torch.float).to(self.device).requires_grad_()
        phi = torch.tensor(self.phi, dtype=torch.float).to(self.device).requires_grad_()

        total_reward = torch.tensor(0, dtype=torch.float).to(self.device)

        for reward_list in reward_list_list:
            cmi_list = [(reward - self.phi_2) / self.phi for reward in reward_list]

            phi_reward = phi * torch.tensor(cmi_list, dtype=torch.float).to(self.device) + phi_2

            total_reward += torch.sum(phi_reward)

        loss = torch.mean(F.mse_loss(commission.detach(), total_reward))

        phi_gradient = torch.autograd.grad(loss, phi, create_graph=True)[0]
        phi_2_gradient = torch.autograd.grad(loss, phi_2, create_graph=True)[0]

        self.phi = max(1e-6, self.phi - max(-1, min(self.cmi_lr * phi_gradient.item(), 1)))
        self.phi_2 = self.phi_2 - max(-1, min(self.cmi_lr * phi_2_gradient.item(), 1))

    def fine_tune(self,
                  commission,
                  reward_list_list,
                  ):
        commission = torch.tensor(self.eta * commission, dtype=torch.float).to(self.device)

        phi_2 = torch.tensor(self.phi_2, dtype=torch.float).to(self.device).requires_grad_()
        phi = torch.tensor(self.phi, dtype=torch.float).to(self.device).requires_grad_()

        total_reward = torch.tensor(0, dtype=torch.float).to(self.device)

        for reward_list in reward_list_list:
            cmi_list = [(reward - self.phi_2) / self.phi for reward in reward_list]

            phi_reward = phi * torch.tensor(cmi_list, dtype=torch.float).to(self.device) + phi_2

            total_reward += torch.sum(phi_reward)

        loss = torch.mean(F.mse_loss(commission.detach(), total_reward))

        phi_gradient = torch.autograd.grad(loss, phi, create_graph=True)[0]
        phi_2_gradient = torch.autograd.grad(loss, phi_2, create_graph=True)[0]

        phi_gradient_2 = torch.autograd.grad(phi_gradient, phi, create_graph=True)[0]
        phi_2_gradient_2 = torch.autograd.grad(phi_2_gradient, phi_2, create_graph=True)[0]

        self.phi = max(1e-6, self.phi - max(-1, min(phi_gradient.item() * (phi_gradient_2.item() ** (-1)), 1)))
        self.phi_2 = self.phi_2 - max(-1, min(phi_2_gradient.item() * (phi_2_gradient_2.item() ** (-1)), 1))


if __name__ == '__main__':
    agent = Agent()
    agent.vae_save(1)
