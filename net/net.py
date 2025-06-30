import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.model_config import vae_config, policy_config, value_config


class VAENet(nn.Module):
    def __init__(self,
                 idle_worker_input_dim=vae_config['idle_worker_input_dim'],
                 idle_worker_hidden_dim=vae_config['idle_worker_hidden_dim'],
                 idle_worker_num_layers=vae_config['idle_worker_num_layers'],
                 idle_worker_num_heads=vae_config['idle_worker_num_heads'],

                 around_worker_input_dim=vae_config['around_worker_input_dim'],
                 around_worker_hidden_dim=vae_config['around_worker_hidden_dim'],
                 around_worker_num_layers=vae_config['around_worker_num_layers'],
                 around_worker_num_heads=vae_config['around_worker_num_heads'],

                 around_task_input_dim=vae_config['around_task_input_dim'],
                 around_task_hidden_dim=vae_config['around_task_hidden_dim'],
                 around_task_num_layers1=vae_config['around_task_num_layers'],
                 around_task_num_heads1=vae_config['around_task_num_heads'],

                 optimal_task_input_dim=vae_config['optimal_task_input_dim'],
                 optimal_task_hidden_dim=vae_config['optimal_task_hidden_dim'],
                 optimal_task_num_layers=vae_config['optimal_task_num_layers'],
                 optimal_task_num_heads=vae_config['optimal_task_num_heads'],

                 dropout=vae_config['dropout'],

                 concat_dim=vae_config['concat_dim'],

                 latent_dim=vae_config['decoder_latent_dim'],
                 hidden_dim=vae_config['decoder_hidden_dim'],
                 output_dim=vae_config['decoder_output_dim']

                 ):
        super(VAENet, self).__init__()

        #     encoder
        self.MHA_idle_w = MHA(idle_worker_input_dim, idle_worker_hidden_dim, idle_worker_num_layers,
                              idle_worker_num_heads, dropout)
        self.MHA_around_w = MHA(around_worker_input_dim, around_worker_hidden_dim, around_worker_num_layers,
                                around_worker_num_heads, dropout)
        self.MHA_around_t = MHA(around_task_input_dim, around_task_hidden_dim, around_task_num_layers1,
                                around_task_num_heads1, dropout)
        self.MHA_optimal_t = MHA(optimal_task_input_dim, optimal_task_hidden_dim, optimal_task_num_layers,
                                 optimal_task_num_heads, dropout)

        self.MLP = MLP(
            idle_worker_hidden_dim + around_worker_hidden_dim + around_task_hidden_dim + optimal_task_hidden_dim + around_worker_input_dim,
            concat_dim * 3, concat_dim)

        self.fc_mu = nn.Linear(concat_dim, latent_dim)
        self.fc_logvar = nn.Linear(concat_dim, latent_dim)

        #     decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def encode(self, idle_worker, around_worker, around_task, optimal_task, self_info, select_task,
               mask_idle_w, mask_around_w, mask_around_t, mask_optimal_t, device="cuda"):
        zero_task = torch.zeros(optimal_task.shape[0], 1, optimal_task.shape[2]).to(device)
        optimal_task = torch.cat((zero_task, optimal_task), dim=-2)

        zero_tensor = torch.zeros(mask_optimal_t.size(0), 1, dtype=mask_optimal_t.dtype).to(device)
        mask_optimal_t = torch.cat([zero_tensor, mask_optimal_t], dim=1)

        idle_worker = self.MHA_idle_w(idle_worker, mask_idle_w)
        around_worker = self.MHA_around_w(around_worker, mask_around_w)
        around_task = self.MHA_around_t(around_task, mask_around_t)
        optimal_task = self.MHA_optimal_t(optimal_task, mask_optimal_t)

        idle_worker = torch.nan_to_num(idle_worker, nan=0.0)
        around_worker = torch.nan_to_num(around_worker, nan=0.0)
        around_task = torch.nan_to_num(around_task, nan=0.0)
        optimal_task = torch.nan_to_num(optimal_task, nan=0.0)

        idle_worker = idle_worker.mean(dim=-2)
        around_worker = around_worker.mean(dim=-2)
        around_task = around_task.mean(dim=-2)
        optimal_task = torch.stack([optimal_task[i][select_task[i]] for i in range(len(optimal_task))])

        x = torch.cat((idle_worker, around_worker, around_task, optimal_task, self_info), dim=-1)
        x = self.MLP(x)
        x = F.relu(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, idle_worker, around_worker, around_task, optimal_task, self_info, select_task,
                mask_idle_w, mask_around_w, mask_around_t, mask_optimal_t, device="cuda"):
        mu, logvar = self.encode(idle_worker, around_worker, around_task, optimal_task, self_info, select_task,
                                 mask_idle_w, mask_around_w, mask_around_t, mask_optimal_t, device)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class PolicyNet(nn.Module):
    def __init__(self,
                 idle_worker_input_dim=policy_config['idle_worker_input_dim'],
                 idle_worker_hidden_dim=policy_config['idle_worker_hidden_dim'],
                 idle_worker_num_layers=policy_config['idle_worker_num_layers'],
                 idle_worker_num_heads=policy_config['idle_worker_num_heads'],

                 around_worker_input_dim=policy_config['around_worker_input_dim'],
                 around_worker_hidden_dim=policy_config['around_worker_hidden_dim'],
                 around_worker_num_layers=policy_config['around_worker_num_layers'],
                 around_worker_num_heads=policy_config['around_worker_num_heads'],

                 around_task_input_dim=policy_config['around_task_input_dim'],
                 around_task_hidden_dim=policy_config['around_task_hidden_dim'],
                 around_task_num_layers1=policy_config['around_task_num_layers'],
                 around_task_num_heads1=policy_config['around_task_num_heads'],

                 optimal_task_input_dim=policy_config['optimal_task_input_dim'],
                 optimal_task_hidden_dim=policy_config['optimal_task_hidden_dim'],
                 optimal_task_num_layers=policy_config['optimal_task_num_layers'],
                 optimal_task_num_heads=policy_config['optimal_task_num_heads'],

                 dropout=policy_config['dropout'],

                 mlp_hidden_dim=policy_config['mlp_hidden_dim']

                 ):
        super(PolicyNet, self).__init__()

        self.optimal_task_input_dim = optimal_task_input_dim

        self.MHA_idle_w = MHA(idle_worker_input_dim, idle_worker_hidden_dim, idle_worker_num_layers,
                              idle_worker_num_heads, dropout)
        self.MHA_around_w = MHA(around_worker_input_dim, around_worker_hidden_dim, around_worker_num_layers,
                                around_worker_num_heads, dropout)
        self.MHA_around_t = MHA(around_task_input_dim, around_task_hidden_dim, around_task_num_layers1,
                                around_task_num_heads1, dropout)
        self.MHA_optimal_t = MHA(optimal_task_input_dim, optimal_task_hidden_dim, optimal_task_num_layers,
                                 optimal_task_num_heads, dropout)

        self.MLP = MLP(
            idle_worker_hidden_dim + around_worker_hidden_dim + around_task_hidden_dim + idle_worker_input_dim,
            optimal_task_hidden_dim * 2, optimal_task_hidden_dim)

        self.decoder = MHA(optimal_task_hidden_dim + optimal_task_hidden_dim, optimal_task_hidden_dim,
                           optimal_task_num_layers, optimal_task_num_heads, dropout)

        self.decoder_mlp = MLP(optimal_task_hidden_dim, mlp_hidden_dim, 1)

        self.softmax = nn.Softmax(dim=-2)

    def forward(self, idle_worker, around_worker, around_task, optimal_task, self_info,
                mask_idle_w, mask_around_w, mask_around_t, mask_optimal_t, device="cuda"):
        zero_task = torch.zeros(optimal_task.shape[0], 1, self.optimal_task_input_dim).to(device)
        optimal_task = torch.cat((zero_task, optimal_task), dim=-2)

        zero_tensor = torch.zeros(mask_optimal_t.size(0), 1, dtype=mask_optimal_t.dtype).to(device)
        mask_optimal_t = torch.cat([zero_tensor, mask_optimal_t], dim=1)

        idle_worker = self.MHA_idle_w(idle_worker, mask_idle_w)
        around_worker = self.MHA_around_w(around_worker, mask_around_w)
        around_task = self.MHA_around_t(around_task, mask_around_t)
        optimal_task = self.MHA_optimal_t(optimal_task, mask_optimal_t)

        idle_worker = idle_worker.mean(dim=-2)
        around_worker = around_worker.mean(dim=-2)
        around_task = around_task.mean(dim=-2)

        idle_worker = torch.nan_to_num(idle_worker, nan=0.0)
        around_worker = torch.nan_to_num(around_worker, nan=0.0)
        around_task = torch.nan_to_num(around_task, nan=0.0)

        x = torch.cat((idle_worker, around_worker, around_task, self_info), dim=-1)
        x = self.MLP(x)
        x = x.unsqueeze(1)

        x = x.repeat(1, optimal_task.shape[1], 1)
        x = torch.cat((optimal_task, x), dim=-1)

        x = self.decoder(x, mask_optimal_t)
        x = self.decoder_mlp(x)

        mask_expanded = mask_optimal_t.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(mask_expanded, -1e9)
        x = self.softmax(x)
        return x


class ValueNet(nn.Module):
    def __init__(self,
                 idle_worker_input_dim=value_config['idle_worker_input_dim'],
                 idle_worker_hidden_dim=value_config['idle_worker_hidden_dim'],
                 idle_worker_num_layers=value_config['idle_worker_num_layers'],
                 idle_worker_num_heads=value_config['idle_worker_num_heads'],

                 around_worker_input_dim=value_config['around_worker_input_dim'],
                 around_worker_hidden_dim=value_config['around_worker_hidden_dim'],
                 around_worker_num_layers=value_config['around_worker_num_layers'],
                 around_worker_num_heads=value_config['around_worker_num_heads'],

                 around_task_input_dim=value_config['around_task_input_dim'],
                 around_task_hidden_dim=value_config['around_task_hidden_dim'],
                 around_task_num_layers1=value_config['around_task_num_layers'],
                 around_task_num_heads1=value_config['around_task_num_heads'],

                 optimal_task_input_dim=value_config['optimal_task_input_dim'],
                 optimal_task_hidden_dim=value_config['optimal_task_hidden_dim'],
                 optimal_task_num_layers=value_config['optimal_task_num_layers'],
                 optimal_task_num_heads=value_config['optimal_task_num_heads'],

                 dropout=value_config['dropout'],

                 mlp_hidden_dim=value_config['mlp_hidden_dim']

                 ):
        super(ValueNet, self).__init__()

        self.optimal_task_input_dim = optimal_task_input_dim

        self.MHA_idle_w = MHA(idle_worker_input_dim, idle_worker_hidden_dim, idle_worker_num_layers,
                              idle_worker_num_heads, dropout)
        self.MHA_around_w = MHA(around_worker_input_dim, around_worker_hidden_dim, around_worker_num_layers,
                                around_worker_num_heads, dropout)
        self.MHA_around_t = MHA(around_task_input_dim, around_task_hidden_dim, around_task_num_layers1,
                                around_task_num_heads1, dropout)
        self.MHA_optimal_t = MHA(optimal_task_input_dim, optimal_task_hidden_dim, optimal_task_num_layers,
                                 optimal_task_num_heads, dropout)

        self.MLP = MLP(
            idle_worker_hidden_dim + around_worker_hidden_dim + around_task_hidden_dim + optimal_task_hidden_dim + idle_worker_input_dim,
            optimal_task_hidden_dim * 3, optimal_task_hidden_dim)
        self.MLP2 = MLP(optimal_task_hidden_dim, mlp_hidden_dim, 1)

    def forward(self, idle_worker, around_worker, around_task, optimal_task, self_info,
                mask_idle_w, mask_around_w, mask_around_t, mask_optimal_t, device="cuda"):

        zero_task = torch.zeros(optimal_task.shape[0], 1, self.optimal_task_input_dim).to(device)
        optimal_task = torch.cat((zero_task, optimal_task), dim=-2)

        zero_tensor = torch.zeros(mask_optimal_t.size(0), 1, dtype=mask_optimal_t.dtype).to(device)

        mask_optimal_t = torch.cat([zero_tensor, mask_optimal_t], dim=1)

        idle_worker = self.MHA_idle_w(idle_worker, mask_idle_w)
        around_worker = self.MHA_around_w(around_worker, mask_around_w)
        around_task = self.MHA_around_t(around_task, mask_around_t)
        optimal_task = self.MHA_optimal_t(optimal_task, mask_optimal_t)

        idle_worker = idle_worker.mean(dim=-2)
        around_worker = around_worker.mean(dim=-2)
        around_task = around_task.mean(dim=-2)
        optimal_task = optimal_task.mean(dim=-2)

        x = torch.cat((idle_worker, around_worker, around_task, optimal_task, self_info), dim=-1)
        x = self.MLP(x)
        x = F.relu(x)
        x = self.MLP2(x)
        return x



class MHA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(MHA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout),
                nn.LayerNorm(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(hidden_dim)
            ]))

    def forward(self, x, padding_mask=None):

        x = self.input_projection(x)

        x = x.permute(1, 0, 2)

        for self_attn, norm1, feed_forward, norm2 in self.layers:
            residual = x
            attn_output, _ = self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=padding_mask
            )
            x = norm1(residual + attn_output)

            residual = x
            ff_output = feed_forward(x)
            x = norm2(residual + ff_output)

        return x.permute(1, 0, 2)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.mlp(x)


if __name__ == "__main__":
    pass
