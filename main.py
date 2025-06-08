import os
import random
import argparse
import yaml
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, knn_graph
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

# --------- 환경/seed reproducibility ---------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# --------- Config 관리 ---------
def load_config(config_path):
    try:
        print(f"Trying to load config from: {config_path}")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError("YAML file is empty or cannot be parsed.")
        print(f"Loaded config: {cfg}")
        return cfg
    except Exception as e:
        print(f"[ERROR] Failed to load config file: {e}")
        return None

class Args:
    config = "config.yaml"
    train = True
    test = True
    
args = Args()

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat)) if bias else None

    def forward(self, x):
        out = x @ self.weight.t()
        if self.bias is not None:
            out += self.bias
        return out

def get_region_mask(coords, target_region_center, target_radius, boundary_range, xy_range, z_range):
    dists = torch.norm(coords - torch.tensor(target_region_center, dtype=coords.dtype, device=coords.device), dim=1)
    target_mask = dists < target_radius
    boundary_mask = (
        (torch.abs(coords[:,0] + xy_range) < boundary_range) | (torch.abs(coords[:,0] - xy_range) < boundary_range) |
        (torch.abs(coords[:,1] + xy_range) < boundary_range) | (torch.abs(coords[:,1] - xy_range) < boundary_range) |
        (torch.abs(coords[:,2] - z_range[0]) < boundary_range) | (torch.abs(coords[:,2] - z_range[1]) < boundary_range)
    )
    region_mask = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
    region_mask[boundary_mask] = 2
    region_mask[target_mask] = 1
    return region_mask

def complex_mse_loss(pred, target):
    return ((pred - target).abs() ** 2).mean()
    
def to_complex(tensor):
    # numpy 배열이 들어오면 Tensor로 변환
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    return torch.complex(tensor[:,0], tensor[:,1])

class FFTLinearOperator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_dim, in_dim, dtype=torch.cfloat) * 0.01)
    def forward(self, x):
        x_c = torch.complex(x, torch.zeros_like(x))
        x_fft = torch.fft.fft(x_c, dim=1)
        x_out = torch.fft.ifft(x_fft * self.kernel, dim=1)
        return torch.view_as_real(x_out)[..., 0]  # 복소수 실수부(혹은 그대로 complex로 써도 됨)

class SoftRegionOperator(nn.Module):
    def __init__(self, in_dim, out_dim, n_ops=3):
        super().__init__()
        self.n_ops = n_ops
        self.op_selector = nn.Sequential(
            nn.Linear(n_ops, 32), nn.ReLU(), nn.Linear(32, n_ops)
        )
        self.operators = nn.ModuleList([
            RegionMLP(in_dim, out_dim),       # 0: MLP
            SimpleFNO3DBlock(in_dim, out_dim),# 1: FNO
            FFTLinearOperator(in_dim, out_dim)# 2: FFT
        ])
    def forward(self, x, region_feat, grid_shape=None):
        op_logits = self.op_selector(region_feat)   # (N, n_ops)
        op_weights = torch.softmax(op_logits, dim=-1)
        outs = []
        for i, op in enumerate(self.operators):
            # FNO만 grid_shape 필요
            if isinstance(op, SimpleFNO3DBlock) and grid_shape is not None:
                outs.append(op(x, grid_shape))
            else:
                outs.append(op(x))
        outs = torch.stack(outs, dim=-1)           # (N, out_dim, n_ops)
        out = (op_weights.unsqueeze(1) * outs).sum(dim=-1)  # (N, out_dim)
        return out

class MultiScaleFourierEncoder(nn.Module):
    def __init__(self, input_dim, scales, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales
        self.out_dim = out_dim
        self.B = nn.Parameter(
            torch.randn(input_dim, out_dim), requires_grad=False
        )  # (input_dim, out_dim)
    def forward(self, x):
        # x: (N, input_dim)
        features = []
        for s in self.scales:
            s = float(s)
            proj = x * s @ self.B  # (N, out_dim)
            features.append(torch.sin(proj))
            features.append(torch.cos(proj))
        # [sin(s1), cos(s1), sin(s2), cos(s2), ...] so 총 out_dim * 2 * len(scales)
        return torch.cat(features, dim=1)


# --------- GNN + FFM Encoder ---------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim=6, ffm_scales=[1,5,10,30], ffm_dim=128, gnn_dim=1024, out_dim=256):
        super().__init__()
        self.ms_ffm = MultiScaleFourierEncoder(3, ffm_scales, ffm_dim)
        self.gnn1 = SAGEConv(in_dim, gnn_dim)
        self.bn1 = nn.LayerNorm(gnn_dim)
        self.gnn2 = SAGEConv(gnn_dim, gnn_dim)
        self.bn2 = nn.LayerNorm(gnn_dim)
        ms_ffm_out_dim = len(ffm_scales) * 2 * ffm_dim  # <-- 반드시 맞게!
        self.fc = nn.Linear(gnn_dim + ms_ffm_out_dim, out_dim)
        self.fc_bn = nn.LayerNorm(out_dim)
    def forward(self, x, edge_index, node_coords, batch):
        h = F.relu(self.bn1(self.gnn1(x, edge_index)))
        h = F.relu(self.bn2(self.gnn2(h, edge_index)))
        h_ffm = self.ms_ffm(node_coords)
        # print("h_ffm:", h_ffm.shape)   # shape 체크!
        h_cat = torch.cat([h, h_ffm], dim=1)
        h_g = global_mean_pool(h_cat, batch)
        return self.fc_bn(self.fc(h_g))

class RegionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimpleFNO3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=8, width=32, n_layers=4):
        super().__init__()
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_layers = n_layers

        self.input_proj = nn.Linear(in_channels, width)
        self.fno_layers = nn.ModuleList([
            FNO3DLayer(width, width, modes) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(width, out_channels)
        self.activation = nn.GELU()

    def forward(self, x, grid_shape):
        # x: (N, in_channels), grid_shape: (nx, ny, nz)
        N = x.size(0)
        x = self.input_proj(x)
        # Reshape to 3D grid: (nx, ny, nz, width)
        x = x.view(*grid_shape, self.width).permute(3,0,1,2).unsqueeze(0)  # (B=1, C, nx, ny, nz)
        for layer in self.fno_layers:
            x = self.activation(layer(x))
        x = x.permute(0,2,3,4,1).reshape(N, self.width)
        x = self.output_proj(x)
        return x  # (N, out_channels)

class FNO3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # Spectral weights
        self.weight = nn.Parameter(torch.randn(
            in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat
        ) * 0.01)
        # Local mixing
        self.w_local = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, C, nx, ny, nz)
        B, C, nx, ny, nz = x.shape
        x_ft = torch.fft.rfftn(x, s=(nx, ny, nz), dim=[2,3,4])
        # Only keep low freq modes
        out_ft = torch.zeros(B, self.out_channels, nx, ny, nz//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:,:,:self.modes,:self.modes,:self.modes] = torch.einsum(
            "bci...,...oijk->bcoijk", x_ft[:,:,:self.modes,:self.modes,:self.modes], self.weight
        )
        x_ifft = torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=[2,3,4])
        # Add local convolution
        x_local = self.w_local(x)
        return x_ifft + x_local

# --------- Neural Operator Decoder (with UQ) ---------
class ComplexDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.real_drop = nn.Dropout(p)
        self.imag_drop = nn.Dropout(p)
    def forward(self, x):
        return torch.complex(self.real_drop(x.real), self.imag_drop(x.imag))

class PARONet_Decoder(nn.Module):
    def __init__(self, coord_dim=3, latent_dim=256, grid_size=17, out_dim=8, dropout_p=0.2,
                 region_params=None):
        super().__init__()
        self.grid_size = grid_size
        self.coord_layer = nn.Linear(coord_dim, latent_dim)
        self.latent_to_field = ComplexLinear(latent_dim * 2, out_dim)
        self.dropout = ComplexDropout(dropout_p)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.cfloat))
        self.region_params = region_params
        self.soft_operator = SoftRegionOperator(latent_dim*2, out_dim, n_ops=3)
        self.n_ops = 3

    def forward(self, coords, latent):
        coords = coords.float()
        coord_feat = self.coord_layer(coords)
        latent_rep = latent.repeat_interleave(coords.shape[0] // latent.shape[0], dim=0)
        combined = torch.cat([coord_feat, latent_rep.real], dim=1)  # (N, latent_dim*2)
        region_mask = get_region_mask(
            coords,
            target_region_center=self.region_params['target_center'],
            target_radius=self.region_params['target_radius'],
            boundary_range=self.region_params['boundary_tol'],
            xy_range=self.region_params['xy_range'],
            z_range=self.region_params['z_range'],
        )
        region_feat = F.one_hot(region_mask, num_classes=self.n_ops).float()  # (N, n_ops)

        # ---- grid_shape 추정 ----
        # 예: 전체 query point 개수 == n_grid^3 일 때
        region_len = coords.shape[0]
        grid_size = int(round(region_len ** (1/3)))
        grid_shape = (grid_size, grid_size, grid_size)

        # soft_operator에 grid_shape 전달
        out = self.soft_operator(combined, region_feat, grid_shape=grid_shape)
        out = out + self.bias
        out = self.dropout(out)
        return out

# --------- Full Meta-PINO-AFC Model ---------
class AcousticPINO3D_UQ(nn.Module):
    def __init__(self, in_dim=6, ffm_scales=[1,5,10,30], ffm_dim=128, gnn_dim=256, dec_dim=512, latent_dim=256, out_dim=8, dropout_p=0.2, grid_size=17, region_params=None):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, ffm_scales, ffm_dim, gnn_dim, out_dim=latent_dim)
        self.decoder = PARONet_Decoder(
            coord_dim=3, latent_dim=latent_dim, out_dim=out_dim,
            region_params=region_params
        )
        self.region_params = region_params

    def forward(self, data):
        model_device = next(self.parameters()).device
        x = data.x.to(model_device)
        edge_index = data.edge_index.to(model_device)
        coords = data.coords.to(model_device)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=model_device)
        node_coords = x[:, :3]
        latent = self.encoder(x, edge_index, node_coords, batch)
        coords = coords.float()
        latent = latent.float()
        output = self.decoder(coords, latent)
        return output
    
# SKPDELayer: PDE + Dirichlet BC 통합 잔차 계산 모듈
class ComplexSKPDELayer(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = torch.tensor(k, dtype=torch.cfloat)  # 파라미터도 cfloat

    def forward(self, model, coords_int, coords_bc, gt_bc, latent):
        coords_int = coords_int.float().clone().requires_grad_(True)
        pred_int = model.decoder(coords_int, latent)  # (N, out_dim), cfloat

        # 예시: 첫 채널(압력)만
        p_val = pred_int[:, 0]
        grad_p = torch.autograd.grad(p_val.sum(), coords_int, create_graph=True, allow_unused=True)[0]
        lap_p = sum([torch.autograd.grad(grad_p[:, i].sum(), coords_int, create_graph=True, allow_unused=True)[0][:, i]
                     for i in range(3)]).view(-1)

        helm_res = lap_p + (self.k**2) * p_val
        loss_pde = complex_mse_loss(helm_res, torch.zeros_like(helm_res))

        pred_bc = model.decoder(coords_bc.type(torch.cfloat), latent)
        bc_val = pred_bc[:, 0]
        loss_bc = complex_mse_loss(bc_val, gt_bc.type(torch.cfloat)[:, 0])
        return loss_pde, loss_bc, helm_res.detach()
    
class RegionOpRLAgent:
    def __init__(self, ops=['fft','fno','mlp'], n_regions=3, epsilon=0.2):
        self.ops = ops
        self.n_regions = n_regions
        self.epsilon = epsilon
        # region x op 평균 reward 관리
        self.reward_table = {r: {op: 0.0 for op in ops} for r in range(n_regions)}
        self.count_table = {r: {op: 1.0 for op in ops} for r in range(n_regions)}  # 1.0 초기화(0 나눔 방지)
    
    def select_operators(self, region_stats=None):
        op_config = {}
        for r in range(self.n_regions):
            if np.random.rand() < self.epsilon:
                op = random.choice(self.ops)
            else:
                # 현재까지 평균 reward가 높은 op 선택
                rewards = self.reward_table[r]
                op = max(rewards, key=lambda o: rewards[o] / self.count_table[r][o])
            op_config[r] = op
        return op_config
    
    def update(self, op_config, rewards):
        # rewards: dict(region: reward)
        for r in range(self.n_regions):
            op = op_config[r]
            self.reward_table[r][op] += rewards.get(r, 0.0)
            self.count_table[r][op] += 1.0

# --------- Data Generation ---------
def generate_transducer_array_3d(n_transducers=64, phase_random=True, amp_base=50.0, amp_var=0.1):
    grid_size = int(round(n_transducers ** (1/3)))
    x = np.linspace(-0.02, 0.02, grid_size)
    y = np.linspace(-0.02, 0.02, grid_size)
    z = np.zeros(grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    if phase_random:
        phases = np.random.uniform(0, 2*np.pi, n_transducers)
    else:
        phases = np.zeros(n_transducers)
    amplitudes = amp_base * (1 + amp_var * np.random.randn(n_transducers))
    return positions, phases, amplitudes

def sample_query_points_3d(grid_size=17, xy_range=0.06, z_range=(0.01, 0.07)):
    if grid_size % 2 == 0:
        grid_size += 1
    x = np.linspace(-xy_range, xy_range, grid_size)
    y = np.linspace(-xy_range, xy_range, grid_size)
    mid = grid_size // 2
    y[mid] = 0.0
    z = np.linspace(z_range[0], z_range[1], grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return coords, X, Y, Z

def synthesize_pressure_and_velocity_3d(positions, phases, amplitudes, coords,
                                        k=2*np.pi/0.0086, omega=2*np.pi*40000, rho0=1.225):
    N, M = positions.shape[0], coords.shape[0]
    diff = coords[None, :, :] - positions[:, None, :]
    dists = np.linalg.norm(diff, axis=2) + 1e-7
    amp = amplitudes[:, None] / dists
    exp_term = np.exp(1j * (k*dists - phases[:, None]))
    field = np.sum(amp * exp_term, axis=0)
    p_real = np.real(field)
    p_imag = np.imag(field)
    grad_p = np.zeros((M, 3), dtype=np.complex128)
    for i in range(3):
        grad_p[:, i] = np.sum(amp * exp_term * (-1) * (diff[:,:,i]/dists) * k, axis=0) / N
    v = -1/(1j*omega*rho0) * grad_p
    v_real = np.real(v)
    v_imag = np.imag(v)
    out = np.concatenate([p_real[:,None], p_imag[:,None], v_real, v_imag], axis=1)
    return out

def build_pyg_data_3d(positions, phases, amplitudes, coords, field, knn=4, device='cpu'):
    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    phases = torch.tensor(phases, dtype=torch.float32, device=device)
    amplitudes = torch.tensor(amplitudes, dtype=torch.float32, device=device)
    node_feat = torch.cat([
        positions, 
        torch.cos(phases).unsqueeze(-1), 
        torch.sin(phases).unsqueeze(-1),
        amplitudes.unsqueeze(-1)
    ], dim=1)
    edge_index = knn_graph(positions, k=knn)
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    y = torch.tensor(field, dtype=torch.float32, device=device) # (N, 8)
    # → coords_batch은 data 생성 시 0으로 설정 (단일 그래프 기준)
    coords_batch = torch.zeros(coords.shape[0], dtype=torch.long, device=device)
    data = Data(x=node_feat, edge_index=edge_index, coords=coords, y=y, coords_batch=coords_batch)
    return data

def mask_boundary_points_3d(coords, xy_range=0.06, z_range=(0.01, 0.07), tol=1e-5):
    mask = (
        (np.abs(coords[:,0] + xy_range) < tol) | (np.abs(coords[:,0] - xy_range) < tol) |
        (np.abs(coords[:,1] + xy_range) < tol) | (np.abs(coords[:,1] - xy_range) < tol) |
        (np.abs(coords[:,2] - z_range[0]) < tol) | (np.abs(coords[:,2] - z_range[1]) < tol)
    )
    return mask

def create_synthetic_dataset_3d(
    n_samples=60,
    n_transducers=64,
    grid_size=10,
    xy_range=0.06,
    z_range=(0.01, 0.07),
    amp_base=50.0,
    amp_var=0.1,
    device='cpu'
):
    data_list = []
    for i in range(n_samples):
        positions, phases, amplitudes = generate_transducer_array_3d(
            n_transducers, amp_base=amp_base, amp_var=amp_var
        )
        coords, _, _, _ = sample_query_points_3d(grid_size, xy_range, z_range)
        field = synthesize_pressure_and_velocity_3d(positions, phases, amplitudes, coords)
        data = build_pyg_data_3d(positions, phases, amplitudes, coords, field, device=device)
        mask_b = mask_boundary_points_3d(coords, xy_range, z_range)
        data.mask_boundary = torch.tensor(mask_b, dtype=torch.bool, device=device)
        data.mask_interior = ~data.mask_boundary
        data.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        data.phases = torch.tensor(phases, dtype=torch.float32, device=device)
        data.amplitudes = torch.tensor(amplitudes, dtype=torch.float32, device=device)
        data.coords_np = coords
        data.y_np = field
        data.k = 2 * np.pi / 0.0086
        data.omega = 2 * np.pi * 40000
        data.rho0 = 1.225
        data.a = 1e-4
        data.f1 = 1.0
        data.f2 = 1.0
        data.xy_range = xy_range
        data.z_range = z_range
        data.grid_size = grid_size
        data.boundary_coords = torch.tensor(coords[mask_b], dtype=torch.float32, device=device)
        data.boundary_p = torch.tensor(field[mask_b, :2], dtype=torch.float32, device=device)
        data_list.append(data)
    return data_list

def custom_collate(data_list):
    return Batch.from_data_list(data_list)

# --------- Target Pressure Loss ---------
def target_point_loss(model, target_coord, latent):
    pred = model.decoder(target_coord.type(torch.cfloat), latent.type(torch.cfloat))
    p_abs = pred.abs()
    return -torch.mean(p_abs)

# --------- Early Stopping 및 로그 ---------
class EarlyStopping:
    def __init__(self, patience=1000, min_delta=1e-3, warmup=1000, window=20):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.stop = False
        self.epoch = 0
        self.warmup = warmup
        self.losses = []
        self.window = window

    def __call__(self, val_loss):
        self.epoch += 1
        self.losses.append(val_loss)
        # warmup 기간엔 무조건 진행
        if self.epoch < self.warmup:
            return
        # 최근 window 만큼 평균으로 판단
        avg_loss = np.mean(self.losses[-self.window:])
        if avg_loss + self.min_delta < self.best_loss:
            self.best_loss = avg_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter > self.patience:
            self.stop = True
            
def plot_region_kernel(model):
    plt.hist(model.decoder.kernel_bg.detach().cpu().abs().numpy().flatten(), bins=40, alpha=0.5, label="background")
    plt.hist(model.decoder.kernel_target.detach().cpu().abs().numpy().flatten(), bins=40, alpha=0.5, label="target")
    plt.hist(model.decoder.kernel_boundary.detach().cpu().abs().numpy().flatten(), bins=40, alpha=0.5, label="boundary")
    plt.legend(); plt.title("Region-wise Operator Kernel Magnitude"); plt.show()

def plot_residual_hist(residuals):
    plt.figure()
    plt.hist(residuals, bins=40, alpha=0.8)
    plt.title("Physics-Informed Residual Distribution")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

# --------- Training ---------
def train_model(model, train_loader, device, y_mean, y_std, target_coord, cfg, region_params, k, epochs, agent=None, lr=1e-5, w_pde=1.0, w_bc=1.0, w_target=1.0, logdir="./logs"):
    w_target_region = cfg.get('w_target_region', 1.0)
    w_boundary_region = cfg.get('w_boundary_region', 1.0)
    w_background_region = cfg.get('w_background_region', 1.0)
    w_grad_penalty = cfg.get('w_grad_penalty', 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, verbose=True)
    early_stop = EarlyStopping(patience=100)
    skpde = ComplexSKPDELayer(k=2*np.pi/0.0086)
    loss_hist = []
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        total_loss = 0
        if agent is not None:
            region_operator_config = agent.select_operators()
            model.decoder.region_operator_config = region_operator_config
        for batch in train_loader:
            pred = model(batch)  # (총 쿼리 수, 8)
            n_graphs = int(batch.coords_batch.max().item()) + 1
            loss_pressure = 0.0
            loss_velocity = 0.0
            for i in range(n_graphs):
                idx = (batch.coords_batch == i)
                pred_i = pred[idx]
                y_i = batch.y[idx]
                y_cpx_i = torch.complex(y_i[:,0], y_i[:,1])
                y_real_i = y_i[:,2:]
                loss_pressure += complex_mse_loss(pred_i[:,0], y_cpx_i)
                loss_velocity += F.mse_loss(pred_i[:,2:].real, y_real_i)
            loss_pressure /= n_graphs
            loss_velocity /= n_graphs
            loss_data = loss_pressure + loss_velocity

            # --- Physics-Informed Loss ---
            coords_int_full = batch.coords[batch.mask_interior].to(device).float()
            coords_int_full.requires_grad_(True)
            region_mask = get_region_mask(
                coords_int_full,
                target_region_center=region_params['target_center'],
                target_radius=region_params['target_radius'],
                boundary_range=region_params['boundary_tol'],
                xy_range=region_params['xy_range'],
                z_range=region_params['z_range']
            )

            latent = model.encoder(
                batch.x, batch.edge_index, batch.x[:, :3],
                torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            )

            # --- Decoder output ---
            pred_all = model.decoder(coords_int_full, latent)
            pred_all_real_imag = torch.view_as_real(pred_all[:, 0])  # (N, 2)
            p_real = pred_all_real_imag[:, 0] * y_std[0] + y_mean[0]
            p_imag = pred_all_real_imag[:, 1] * y_std[1] + y_mean[1]

            # 1차 미분(벡터)
            grad_real = torch.autograd.grad(p_real, coords_int_full, grad_outputs=torch.ones_like(p_real),
                                           create_graph=True, retain_graph=True)[0]  # (N, 3)
            grad_imag = torch.autograd.grad(p_imag, coords_int_full, grad_outputs=torch.ones_like(p_imag),
                                           create_graph=True, retain_graph=True)[0]  # (N, 3)

            # 2차 미분(라플라시안)
            lap_real = torch.zeros_like(p_real)
            lap_imag = torch.zeros_like(p_imag)
            for i in range(3):
                grad2_real = torch.autograd.grad(grad_real[:, i], coords_int_full, grad_outputs=torch.ones_like(grad_real[:, i]),
                                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
                grad2_imag = torch.autograd.grad(grad_imag[:, i], coords_int_full, grad_outputs=torch.ones_like(grad_imag[:, i]),
                                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
                if grad2_real is None:
                    print(f"grad2_real is None at i={i}")
                else:
                    lap_real += grad2_real[:, i]
                if grad2_imag is None:
                    print(f"grad2_imag is None at i={i}")
                else:
                    lap_imag += grad2_imag[:, i]

            # PDE residual 계산 (topk 없이 전체)
            helm_res = (lap_real + (k**2) * p_real)**2 + (lap_imag + (k**2) * p_imag)**2
            loss_pde = helm_res.mean()

            # Boundary loss 계산
            pred_bc = model.decoder(batch.boundary_coords.float(), latent)
            bc_val_real_imag = torch.view_as_real(pred_bc[:, 0])
            bc_real = bc_val_real_imag[:, 0]
            bc_imag = bc_val_real_imag[:, 1]
            bc_pred = torch.complex(bc_real, bc_imag)
            bc_true = torch.complex(batch.boundary_p[:, 0], batch.boundary_p[:, 1]).to(device)
            loss_bc = complex_mse_loss(bc_pred, bc_true)

            # 목표 좌표 압력 loss 계산
            t_coord = torch.tensor(target_coord, dtype=torch.float32, device=device).unsqueeze(0)
            loss_target = target_point_loss(model, t_coord, latent)

            # 데이터 loss는 이전과 동일
            loss_data = loss_pressure + loss_velocity

            # 최종 손실
            # region별 손실 계산
            loss_target_region = -pred_all[region_mask==1].abs().mean() if (region_mask==1).sum() > 0 else 0.0
            loss_boundary_region = complex_mse_loss(pred_all[region_mask==2], torch.zeros_like(pred_all[region_mask==2])) if (region_mask==2).sum() > 0 else 0.0
            loss_background_region = (pred_all[region_mask==0].abs()**2).mean() if (region_mask==0).sum() > 0 else 0.0

            if (region_mask==1).sum() > 0:
                idxs = (region_mask==1).nonzero(as_tuple=True)[0]
                try:
                    grad_trap = torch.autograd.grad(
                        pred_all[idxs].abs().sum(), coords_int_full[idxs], create_graph=True, allow_unused=True
                    )[0]
                    grad_penalty = grad_trap.norm(dim=1).mean() if grad_trap is not None else torch.tensor(0.0, device=pred_all.device)
                except Exception as e:
                    print(f"Grad penalty autograd error: {e}")
                    grad_penalty = torch.tensor(0.0, device=pred_all.device)
            else:
                grad_penalty = torch.tensor(0.0, device=pred_all.device)

            # 최종 손실식
            total = (
                    loss_data
                    + w_pde * loss_pde
                    + w_bc * loss_bc
                    + w_target * loss_target
                    + w_target_region * loss_target_region
                    + w_boundary_region * loss_boundary_region
                    + w_background_region * loss_background_region
                    + w_grad_penalty * grad_penalty
                )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_loss += total.item()
        avg_loss = total_loss / len(train_loader)
        loss_hist.append(avg_loss)
        scheduler.step(avg_loss)
        helm_res_used = helm_res.detach()
        if epoch % 500 == 0:
            print(f"[Epoch {epoch}] Total: {avg_loss:.3e} | Data: {loss_data.item():.3e} | PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e} | Target: {loss_target.item():.3e}")
            plot_residual_hist(helm_res_used.cpu().numpy().flatten())
        early_stop(avg_loss)
        if early_stop.stop:
            print(f"Early stopping at epoch {epoch}.")
            break
        if agent is not None:
            reward_dict = {
                0: -loss_background_region.item() if isinstance(loss_background_region, torch.Tensor) else -loss_background_region,
                1: -loss_target_region.item() if isinstance(loss_target_region, torch.Tensor) else -loss_target_region,
                2: -loss_boundary_region.item() if isinstance(loss_boundary_region, torch.Tensor) else -loss_boundary_region,
            }
            agent.update(region_operator_config, reward_dict)
    # Save loss curve
    os.makedirs(logdir, exist_ok=True)
    np.save(os.path.join(logdir, "loss_history.npy"), np.array(loss_hist))
    return loss_hist

# --------- Evaluation (Dropout-UQ) ---------
def set_dropout_train_batchnorm_eval(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, nn.BatchNorm1d):
            m.eval()

def predict_mc_dropout(model, data, n_samples=32, device='cpu'):
    set_dropout_train_batchnorm_eval(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(data.to(device)).cpu().numpy()
            preds.append(pred)
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

def gorkov_potential_torch_3d(p_real, p_imag, vx_r, vy_r, vz_r, vx_i, vy_i, vz_i, a=1e-4, f1=1.0, f2=1.0, rho0=1.225):
    p2 = p_real**2 + p_imag**2
    v2 = vx_r**2 + vy_r**2 + vz_r**2 + vx_i**2 + vy_i**2 + vz_i**2
    U = (4/3)*np.pi*a**3 * (0.5*f1*p2 - 0.75*f2*rho0*v2)
    return U

def plot_field_gorkov_3d(U_mean, U_std, X, Y, Z, title='Gor’kov U (slice)', z_idx=8, savedir=None):
    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    plt.title('Mean Gor’kov U')
    plt.contourf(X[:,:,z_idx], Y[:,:,z_idx], U_mean[:,:,z_idx], levels=50, cmap='viridis')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('Std (Uncertainty)')
    plt.contourf(X[:,:,z_idx], Y[:,:,z_idx], U_std[:,:,z_idx], levels=30, cmap='magma')
    plt.colorbar()
    plt.suptitle(title)
    plt.tight_layout()
    if savedir:
        plt.savefig(os.path.join(savedir, f"gorkov_U_slice{z_idx}.png"))
    plt.show()

# --------- Main 전체 파이프라인 ---------
def main(args):
    # 1. 환경설정 및 config 로드
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 2. 데이터 생성 및 정규화
    print('Generating 3D training data...')
    train_data = create_synthetic_dataset_3d(
        n_samples=cfg['n_train'], n_transducers=cfg['n_transducers'],
        grid_size=cfg['grid_size_train'], amp_base=cfg['amp_base'], device=device
    )
    test_data = create_synthetic_dataset_3d(
        n_samples=cfg['n_test'], n_transducers=cfg['n_transducers'],
        grid_size=cfg['grid_size_test'], amp_base=cfg['amp_base'], device=device
    )
    all_y = np.vstack([d.y_np for d in train_data])
    y_mean = all_y.mean(axis=0)
    y_std = all_y.std(axis=0) + 1e-8
    agent = RegionOpRLAgent(ops=['fft','fno','mlp'], n_regions=3, epsilon=0.2)
    for d in train_data:
        d.y = to_complex(d.y.cpu().numpy())
        d.boundary_p = to_complex(d.boundary_p.cpu().numpy())
    for d in test_data:
        d.y = to_complex(d.y.cpu().numpy())
        d.boundary_p = to_complex(d.boundary_p.cpu().numpy())
    normalization_stats = (y_mean, y_std)
    # custom_collate 함수는 이미 위에서 작성한 대로 사용
    train_loader = DataLoader(
        train_data,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate,
        drop_last=True
    )
    # 3. 모델 초기화
    print('Initializing model...')
    
    def to_float(x):
        if isinstance(x, str):
            return float(x)
        return float(x)

    def to_tuple_float(x):
        if isinstance(x, str):
            x = x.strip("[]()")
            return tuple(map(float, x.split(',')))
        elif isinstance(x, (list, tuple, np.ndarray)):
            return tuple(map(float, x))
        else:
            return (float(x), float(x))
        
    region_params = {
        'target_center': to_tuple_float(cfg.get('target_coord', [0.0, 0.0, 0.04])),
        'target_radius': to_float(cfg.get('target_radius', 0.005)),
        'boundary_tol': to_float(cfg.get('boundary_tol', 1e-5)),
        'xy_range': to_float(cfg.get('xy_range', 0.06)),
        'z_range': to_tuple_float(cfg.get('z_range', (0.01, 0.07))),
    }
    # train_model 호출
    print("region_params", region_params)
    for k, v in region_params.items():
        print(f"{k}: {v} ({type(v)})")
    # AcousticPINO3D_UQ에 region_params 전달!
    # ----------- target_coord 정의 먼저! -----------
    target_coord = cfg.get('target_coord', [0.0, 0.0, 0.04])
    region_operator_config = {0: 'fft', 1: 'fno', 2: 'mlp'}
    model = AcousticPINO3D_UQ(
        in_dim=6, ffm_scales=cfg['ffm_scales'], ffm_dim=cfg['ffm_dim'], gnn_dim=cfg['gnn_dim'],
        dec_dim=cfg['dec_dim'], out_dim=8, dropout_p=cfg['dropout_p'],
        latent_dim=256, grid_size=cfg['grid_size_train'],
        region_params=region_params
    ).to(device)
    agent = RegionOpRLAgent(ops=['fft','fno','mlp'], n_regions=3, epsilon=0.2)
    loss_hist = train_model(
        model, train_loader, device, y_mean, y_std, target_coord, cfg,
        region_params, # <-- 추가
        k=cfg['k'],
        epochs=cfg['epochs'], lr=cfg['lr'], w_pde=cfg['w_pde'], w_bc=cfg['w_bc'], w_target=cfg['w_target'],
        logdir=cfg.get('logdir','./logs'), agent=agent # <-- agent 추가!
    )
    # 4. 학습
    if args.train:
        print('Training (Data + Physics-Informed Loss)...')
        target_coord = cfg.get('target_coord', [0.0, 0.0, 0.04])
        loss_hist = train_model(
            model, train_loader, device, y_mean, y_std, target_coord, cfg,
            epochs=cfg['epochs'], lr=cfg['lr'], w_pde=cfg['w_pde'], w_bc=cfg['w_bc'], w_target=cfg['w_target'], logdir=cfg.get('logdir','./logs')
        )
        save_model(model, os.path.join(cfg.get('logdir','./logs'), 'best_model.pt'))
        plt.plot(loss_hist)
        plt.xlabel('Epoch'); plt.ylabel('Total Loss')
        plt.yscale('log'); plt.title('Training loss')
        plt.savefig(os.path.join(cfg.get('logdir','./logs'), "loss_curve.png"))
        plt.show()

    # 5. 테스트 및 시각화
    if args.test:
        print('Evaluating test data and visualizing...')
        load_model(model, os.path.join(cfg.get('logdir','./logs'), 'best_model.pt'), device)
        model.eval()
        with torch.no_grad():
            for idx, test_batch in enumerate(test_loader):
                test_batch = test_batch.to(device)
                mean_pred, std_pred = predict_mc_dropout(model, test_batch, n_samples=cfg['n_mc'], device=device)
                coords, X, Y, Z = sample_query_points_3d(grid_size=cfg['grid_size_test'])
                y_mean, y_std = normalization_stats
                mean_pred_denorm = mean_pred
                std_pred_denorm = std_pred * y_std
                p_real, p_imag = mean_pred_denorm.real, mean_pred_denorm.imag
                vx_r, vy_r, vz_r, vx_i, vy_i, vz_i = mean_pred_denorm[:,2], mean_pred_denorm[:,3], mean_pred_denorm[:,4], mean_pred_denorm[:,5], mean_pred_denorm[:,6], mean_pred_denorm[:,7]
                U_mean = gorkov_potential_torch_3d(
                    torch.tensor(p_real), torch.tensor(p_imag),
                    torch.tensor(vx_r), torch.tensor(vy_r), torch.tensor(vz_r),
                    torch.tensor(vx_i), torch.tensor(vy_i), torch.tensor(vz_i)
                ).numpy().reshape(cfg['grid_size_test'],cfg['grid_size_test'],cfg['grid_size_test'])
                U_std = np.zeros_like(U_mean)
                for i in range(mean_pred.shape[0]):
                    std_samp = std_pred_denorm[i]
                    U_std.flat[i] = np.linalg.norm(std_samp)
                U_std = U_std.reshape(cfg['grid_size_test'],cfg['grid_size_test'],cfg['grid_size_test'])
                plot_field_gorkov_3d(U_mean, U_std, X, Y, Z, z_idx=cfg['grid_size_test']//2, savedir=cfg.get('logdir','./logs'))
main(args)
# --------- CLI 실행 ---------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta-PINO-AFC 논문 전체 실험 파이프라인")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args, unknown = parser.parse_known_args()
    main(args)
