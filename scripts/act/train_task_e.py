ALGO_NAME = 'BC_ACT'

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers.training_utils import EMAModel

from atec_rl_lab.train.act.act.detr.backbone import build_backbone
from atec_rl_lab.train.act.act.detr.transformer import build_transformer
from atec_rl_lab.train.act.act.detr.detr_vae import build_encoder, DETRVAE
from atec_rl_lab.train.act.act.utils import IterationBasedBatchSampler, worker_init_fn
import tyro


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ATEC2026"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    demo_path: str = './datasets/atec_task_e/trajectory.hdf5'
    """path to the HDF5 demo dataset produced by collect_demos_task_e.py"""
    num_demos: Optional[int] = None
    """number of trajectories to load (None = all)"""
    total_iters: int = 1_000_000
    """total training iterations"""
    batch_size: int = 256
    """batch size"""

    # ACT specific
    lr: float = 1e-4
    """learning rate"""
    kl_weight: float = 10
    """weight for the KL loss term"""
    temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed at inference"""

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False
    """always False — depth not collected; kept for backbone API compatibility"""
    include_rgb: bool = True
    """use RGB images as input (requires --save_images during collection)"""

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    log_freq: int = 1000
    """frequency of logging training metrics"""
    save_freq: int = 5000
    """frequency of saving model checkpoints"""
    num_dataload_workers: int = 0
    """number of DataLoader worker processes"""


class DemoDataset_ACT(Dataset):
    """Load IsaacLab Task-E HDF5 demos into memory.

    HDF5 structure (produced by collect_demos_task_e.py):
        traj_N/obs          (T, 8)       joint positions (qpos)
        traj_N/actions      (T, 8)       env actions
        traj_N/images/rgb   (T, H, W, 3) uint8 RGB — optional
    """

    def __init__(self, data_path: str, num_queries: int,
                 num_traj: Optional[int] = None, include_rgb: bool = True):
        self.num_queries = num_queries
        self.include_rgb = include_rgb
        self.transforms = T.Resize((224, 224), antialias=True)

        # load raw data
        states_list: list[torch.Tensor] = []
        actions_list: list[torch.Tensor] = []
        rgb_list: list[torch.Tensor] = []   # only when include_rgb is True
        has_images = None

        with h5py.File(data_path, 'r') as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
            if num_traj is not None:
                traj_keys = traj_keys[:num_traj]

            for key in traj_keys:
                grp = f[key]
                states_list.append(torch.from_numpy(grp['obs'][:].astype(np.float32)))
                actions_list.append(torch.from_numpy(grp['actions'][:].astype(np.float32)))

                if include_rgb:
                    if has_images is None:
                        has_images = 'images' in grp
                    if has_images and 'images' in grp:
                        rgb_arr = grp['images/rgb'][:]          # (T, H, W, 3) uint8
                        rgb_t = torch.from_numpy(rgb_arr)       # uint8
                        # (T, 3, H, W) → resize → (T, 3, 224, 224)
                        rgb_t = self.transforms(rgb_t.permute(0, 3, 1, 2))
                        # add camera dim → (T, 1, 3, 224, 224)
                        rgb_list.append(rgb_t.unsqueeze(1))

        if has_images is None:
            has_images = False
        self.has_images = has_images and include_rgb and len(rgb_list) > 0

        if include_rgb and not self.has_images:
            print('[WARN] include_rgb=True but no images found in dataset. '
                  'Re-collect with --save_images, or set include_rgb=False.')

        self.num_traj = len(states_list)
        self.states  = states_list   # list of (T, 8)
        self.actions = actions_list  # list of (T, 8)
        self.rgb     = rgb_list      # list of (T, 1, 3, 224, 224) or empty

        # state/action dims
        self.state_dim = self.states[0].shape[1]
        self.act_dim   = self.actions[0].shape[1]

        # index slices: (traj_idx, timestep)
        self.slices = [
            (i, t)
            for i, acts in enumerate(self.actions)
            for t in range(acts.shape[0])
        ]
        print(f'Loaded {self.num_traj} trajectories, {len(self.slices)} timesteps. '
              f'state_dim={self.state_dim}, act_dim={self.act_dim}, '
              f'has_images={self.has_images}')

        # normalisation stats (pd_joint_pos = absolute actions → normalise)
        self.norm_stats = self._compute_norm_stats()

    # ------------------------------------------------------------------

    def _pad_action(self, act_seq: torch.Tensor) -> torch.Tensor:
        """Pad a short action chunk by repeating the last action."""
        shortage = self.num_queries - act_seq.shape[0]
        if shortage > 0:
            act_seq = torch.cat([act_seq, act_seq[-1:].repeat(shortage, 1)], dim=0)
        return act_seq

    def _compute_norm_stats(self) -> dict:
        # Vectorised: stack each full trajectory then slice — avoids 110k tiny ops
        all_states  = torch.cat(self.states,  dim=0)   # (total_T, state_dim)
        all_actions = torch.cat(self.actions, dim=0)   # (total_T, act_dim)

        state_mean = all_states.mean(0,  keepdim=True)
        state_std  = all_states.std(0,   keepdim=True).clamp(1e-2)
        act_mean   = all_actions.mean(0, keepdim=True)
        act_std    = all_actions.std(0,  keepdim=True).clamp(1e-2)

        return dict(state_mean=state_mean, state_std=state_std,
                    action_mean=act_mean,  action_std=act_std)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        state   = self.states[traj_idx][ts]
        act_seq = self._pad_action(self.actions[traj_idx][ts:ts + self.num_queries])

        # normalise
        state   = (state   - self.norm_stats['state_mean'][0])  / self.norm_stats['state_std'][0]
        act_seq = (act_seq - self.norm_stats['action_mean'])     / self.norm_stats['action_std']

        obs = dict(state=state)
        if self.has_images:
            obs['rgb'] = self.rgb[traj_idx][ts]   # (1, 3, 224, 224) uint8

        return {'observations': obs, 'actions': act_seq}



class Agent(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, args: Args):
        super().__init__()
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.kl_weight  = args.kl_weight
        self.normalize  = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        self.include_rgb = args.include_rgb

        # CNN backbone — None for state-only mode (DETRVAE handles both paths)
        backbones = [build_backbone(args)] if args.include_rgb else None

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            action_dim=act_dim,
            num_queries=args.num_queries,
        )

    def _preprocess_rgb(self, obs: dict) -> None:
        if self.include_rgb and 'rgb' in obs:
            obs['rgb'] = obs['rgb'].float() / 255.0
            # obs['rgb']: (B, num_cams, 3, 224, 224)
            B, N, C, H, W = obs['rgb'].shape
            obs['rgb'] = self.normalize(obs['rgb'].view(B * N, C, H, W)).view(B, N, C, H, W)

    def _model_input(self, obs: dict):
        # DETRVAE state-only path expects the state tensor directly, not a dict
        return obs if self.include_rgb else obs['state']

    def compute_loss(self, obs: dict, action_seq: torch.Tensor) -> dict:
        self._preprocess_rgb(obs)
        a_hat, (mu, logvar) = self.model(self._model_input(obs), action_seq)

        total_kld, _, _ = kl_divergence(mu, logvar)
        l1 = F.l1_loss(action_seq, a_hat)

        return dict(l1=l1, kl=total_kld[0],
                    loss=l1 + total_kld[0] * self.kl_weight)

    def get_action(self, obs: dict) -> torch.Tensor:
        self._preprocess_rgb(obs)
        a_hat, _ = self.model(self._model_input(obs))
        return a_hat


def kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu     = mu.view(mu.size(0),     mu.size(1))
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds       = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld  = klds.sum(1).mean(0, True)
    dim_kld    = klds.mean(0)
    mean_kld   = klds.mean(1).mean(0, True)
    return total_kld, dim_kld, mean_kld


def save_ckpt(run_name: str, tag: str) -> None:
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'agent':      agent.state_dict(),
        'ema_agent':  ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')
    print(f'[INFO] Saved checkpoint: runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == '__main__':
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[:-len('.py')]
        run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # dataset & dataloader
    dataset = DemoDataset_ACT(
        args.demo_path,
        num_queries=args.num_queries,
        num_traj=args.num_demos,
        include_rgb=args.include_rgb,
    )
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    sampler       = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=args.seed),
    )

    # logging
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            group='ACT',
            tags=['act'],
        )
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n' +
        '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()),
    )

    # agent
    agent     = Agent(dataset.state_dim, dataset.act_dim, args).to(device)
    ema_agent = Agent(dataset.state_dim, dataset.act_dim, args).to(device)

    param_dicts = [
        {'params': [p for n, p in agent.named_parameters()
                    if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in agent.named_parameters()
                    if 'backbone'     in n and p.requires_grad],
         'lr': args.lr_backbone},
    ]
    optimizer    = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    lr_drop      = int(2 / 3 * args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)
    ema          = EMAModel(parameters=agent.parameters(), power=0.75)

    # training loop
    agent.train()
    best_loss = float('inf')
    timings   = defaultdict(float)

    for cur_iter, data_batch in enumerate(train_dataloader):
        last_tick = time.time()

        obs_batch = {k: v.to(device, non_blocking=True)
                     for k, v in data_batch['observations'].items()}
        act_batch = data_batch['actions'].to(device, non_blocking=True)

        loss_dict  = agent.compute_loss(obs=obs_batch, action_seq=act_batch)
        total_loss = loss_dict['loss']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(agent.parameters())

        timings['update'] += time.time() - last_tick

        if cur_iter % args.log_freq == 0:
            loss_val = total_loss.item()
            print(f'Iter {cur_iter:7d}  loss={loss_val:.4f}  '
                  f'l1={loss_dict["l1"].item():.4f}  '
                  f'kl={loss_dict["kl"].item():.4f}')
            writer.add_scalar('charts/lr',          optimizer.param_groups[0]['lr'], cur_iter)
            writer.add_scalar('charts/lr_backbone', optimizer.param_groups[1]['lr'], cur_iter)
            writer.add_scalar('losses/total',       loss_val,                        cur_iter)
            writer.add_scalar('losses/l1',          loss_dict['l1'].item(),          cur_iter)
            writer.add_scalar('losses/kl',          loss_dict['kl'].item(),          cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f'time/{k}', v, cur_iter)

            if loss_val < best_loss:
                best_loss = loss_val
                save_ckpt(run_name, 'best_loss')

        if args.save_freq > 0 and cur_iter % args.save_freq == 0 and cur_iter > 0:
            save_ckpt(run_name, str(cur_iter))

    save_ckpt(run_name, 'final')
    writer.close()
    print(f'[INFO] Training done. Run: runs/{run_name}')
