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

os.environ["PYTORCH_ENABLE_FLASH_ATTENTION"] = "0"
os.environ["PYTORCH_ENABLE_MEM_EFFICIENT_ATTENTION"] = "0"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"
os.environ["PYTORCH_USE_CUDA_DSA"] = "0"  # DSA(Direct Storage Attention) 비활성화

REGION_OPS = ['fft', 'fno', 'mlp', 'psdtrans']
N_REGION = len(REGION_OPS)

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

def parse_config(cfg):
    """config.yaml에서 읽은 딕셔너리 cfg를 안전하게 형변환/검증."""
    # 숫자
    float_keys = [
        'amp_base', 'xy_range', 'boundary_tol', 'target_radius', 'k', 'lr', 'dropout_p'
    ]
    int_keys = [
        'seed', 'n_train', 'n_test', 'n_transducers', 'grid_size_train', 'grid_size_test',
        'ffm_dim', 'gnn_dim', 'latent_dim', 'dec_dim', 'epochs', 'batch_size', 'n_mc'
    ]
    list_keys = [
        'ffm_scales'
    ]
    tuple_keys = [
        'target_coord', 'z_range'
    ]
    for k in float_keys:
        if k in cfg:
            cfg[k] = float(cfg[k])
    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])
    for k in list_keys:
        if k in cfg and isinstance(cfg[k], str):
            # "1, 5, 10" -> [1, 5, 10]
            cfg[k] = [int(s.strip()) for s in cfg[k].strip('[]').split(',')]
    for k in tuple_keys:
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = tuple(float(s.strip()) for s in cfg[k].strip('[]()').split(','))
    return cfg
    
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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel = nn.Parameter(torch.randn(out_dim, in_dim, dtype=torch.cfloat) * 0.01)
    def forward(self, x):
        # x: (N, in_dim)
        x_c = torch.complex(x, torch.zeros_like(x)) if not torch.is_complex(x) else x
        x_fft = torch.fft.fft(x_c, dim=1)  # (N, in_dim)
        # (N, in_dim) @ (in_dim, out_dim) = (N, out_dim)
        x_fft_out = x_fft @ self.kernel.t()  # (N, out_dim)
        x_out = torch.fft.ifft(x_fft_out, dim=1)  # (N, out_dim)
        return torch.view_as_real(x_out)[..., 0]  # (N, out_dim)

# =========================================================================================
# 1. SoftRegionOperator 클래스를 아래 PARONetOperatorSelector 클래스로 대체
# =========================================================================================
class PARONetOperatorSelector(nn.Module):
    """
    강화학습 에이전트 또는 설정에 따라 지역별로 지정된 단일 연산자(Operator)를
    선택하고 적용하는 '하드 셀렉터(Hard Selector)' 모듈.
    """
    def __init__(self, in_dim, out_dim, n_ops=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 사용 가능한 모든 운영자들을 ModuleList에 등록
        self.operators = nn.ModuleList([
            RegionMLP(in_dim, out_dim),           # 0: 'mlp'
            SimpleFNO3DBlock(in_dim, out_dim),    # 1: 'fno'
            FFTLinearOperator(in_dim, out_dim),   # 2: 'fft'
            PSDTransformerNO(in_dim, out_dim)     # 3: 'psdtrans'
        ])
        
        # 운영자 이름(str)을 ModuleList의 인덱스(int)로 변환하기 위한 맵
        self.op_name_to_idx = {'mlp': 0, 'fno': 1, 'fft': 2, 'psdtrans': 3}

    def forward(self, x, coords, pde_params, op_config, region_params):
        """
        Args:
            x (Tensor): 입력 피처 (N, in_dim)
            coords (Tensor): 쿼리 포인트 좌표 (N, 3)
            pde_params (Tensor): 각 포인트에 매핑된 PDE 파라미터 (N, pde_dim)
            op_config (dict): RL 에이전트가 선택한 지역-운영자 매핑. 예: {0: 'fft', 1: 'fno', 2: 'mlp'}
            region_params (dict): 지역 마스크 생성을 위한 파라미터 딕셔너리.
        """
        # op_config가 없으면, 가장 간단한 MLP를 기본 운영자로 사용 (Fallback)
        if op_config is None:
            return self.operators[0](x)

        # 1. 입력 좌표를 기반으로 각 포인트의 지역(region)을 결정하는 마스크 생성
        region_mask = get_region_mask(
            coords,
            target_region_center=region_params['target_center'],
            target_radius=region_params['target_radius'],
            boundary_range=region_params['boundary_tol'],
            xy_range=region_params['xy_range'],
            z_range=region_params['z_range'],
        )

        # 2. 최종 출력을 담을 텐서 초기화
        outputs = torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=torch.float32)
        
        # 입력 x를 복소수로 변환 (필요시)
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        # 3. 각 지역별로 루프를 돌며 선택된 운영자를 적용
        for region_idx, op_name in op_config.items():
            # 현재 지역에 해당하는 포인트들의 마스크
            mask = (region_mask == region_idx)
            
            # 해당 지역에 포인트가 없으면 다음 지역으로
            if not torch.any(mask):
                continue
            
            # 선택된 운영자 인스턴스 가져오기
            op_idx = self.op_name_to_idx[op_name]
            op = self.operators[op_idx]

            # 운영자별 입력 조건에 맞게 호출
            # FNO와 PSDTransformer는 전체가 정규 격자(grid)일 때 가장 잘 작동합니다.
            # 포인트의 일부만 마스킹된 비정규적 데이터에 대해서는 한계가 있을 수 있으므로,
            # 이 경우 가장 강건한 MLP로 대체(fallback)하는 것이 안정적일 수 있습니다.
            # 여기서는 구현의 편의를 위해 일단 그대로 적용합니다.
            
            x_masked = x[mask]
            
            if op_name == 'fno':
                # FNO는 grid_shape이 필요. 마스킹된 데이터는 정확한 grid를 형성하지 못할 수 있음.
                # 전체 데이터가 하나의 지역일 때 가장 잘 작동함.
                grid_size_approx = int(round(x_masked.shape[0] ** (1/3)))
                grid_shape_approx = (grid_size_approx, grid_size_approx, grid_size_approx)
                if np.prod(grid_shape_approx) == x_masked.shape[0]:
                    result = op(x_masked, grid_shape_approx)
                else:
                    # 격자 형성이 안되면 MLP로 대체
                    result = self.operators[0](x_masked)

            elif op_name == 'psdtrans':
                pde_params_masked = pde_params[mask]
                result = op(x_masked, pde_params=pde_params_masked)

            else: # 'mlp' 또는 'fft' (포인트별 연산)
                result = op(x_masked)

            # 연산 결과를 최종 출력 텐서의 해당 위치에 저장
            # 결과가 복소수일 경우 실수부만 취하거나, out_dim에 맞게 처리 필요.
            # 여기서는 out_dim이 8(실수)이라고 가정하고, 복소수 결과는 실수/허수부로 분리.
            if torch.is_complex(result):
                # 예: p_real, p_imag, v_real, v_imag...
                # out_dim이 8이므로 (N, 4) 복소수 또는 (N, 8) 실수가 나와야 함
                # 간단하게 실수부만 취하거나, 복소수를 실수로 변환
                # 아래는 out_dim=8을 가정하고 real/imag을 합치는 예시
                if result.shape[-1] * 2 == self.out_dim:
                     outputs[mask] = torch.cat([result.real, result.imag], dim=-1)
                else: # 차원이 안맞으면 실수부만 사용
                     outputs[mask] = result.real
            else:
                 outputs[mask] = result

        # PARONet_Decoder는 복소수 출력을 기대하므로, 최종 결과를 복소수로 변환
        # out_dim이 8이므로, 앞 4개는 real, 뒤 4개는 imag로 가정
        out_real = outputs[:, :self.out_dim//2]
        out_imag = outputs[:, self.out_dim//2:]
        return torch.complex(out_real, out_imag)
    
class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert embed_dim % n_heads == 0
        self.head_dim = embed_dim // n_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, N, embed_dim)
        B, N, D = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        # Split heads
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1,2)  # (B, nH, N, dH)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1,2)
        # Attention score
        attn_score = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim**0.5)  # (B, nH, N, N)
        attn = F.softmax(attn_score, dim=-1)
        out = torch.matmul(attn, v) # (B, nH, N, dH)
        out = out.transpose(1,2).contiguous().view(B, N, D)
        return self.out_proj(out)

class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SimpleSelfAttention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim*mlp_ratio, embed_dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
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
        self.out_dim = out_dim
    def forward(self, x):
        return self.net(x)

class SimpleFNO3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=8, width=32, n_layers=4):
        super().__init__()
        self.width = width
        self.input_proj = nn.Linear(in_channels, width)
        self.fno_layers = nn.ModuleList([
            FNO3DLayer(width, width, modes) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(width, out_channels)
        self.activation = nn.GELU()
        self.out_dim = out_channels

    def forward(self, x, grid_shape):
        N = x.size(0)
        x = self.input_proj(x)  # (N, width)
        x = x.view(*grid_shape, self.width).permute(3,0,1,2).unsqueeze(0)  # (1, width, nx, ny, nz)
        for layer in self.fno_layers:
            x = self.activation(layer(x))
        x = x.permute(0,2,3,4,1).reshape(N, self.width)
        x = self.output_proj(x)
        return x

class FNO3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.w_local = nn.Conv3d(in_channels, out_channels, 1)
        self.weight = nn.Parameter(
            torch.randn(
                in_channels, out_channels,
                modes, modes, modes,
                dtype=torch.cfloat
            ) * 0.01
        )

    def forward(self, x):
        B, C, nx, ny, nz = x.shape
        modes_x = min(self.modes, nx)
        modes_y = min(self.modes, ny)
        modes_z = nz // 2 + 1
        modes_z_eff = min(self.modes, modes_z)

        x_ft = torch.fft.rfftn(x, s=(nx, ny, nz), dim=[2,3,4])
        # 디버그 출력
        print(f"x_ft.shape: {x_ft.shape}")
        print(f"weight.shape (slice): {self.weight[:, :, :modes_x, :modes_y, :modes_z_eff].shape}")

        # 슬라이스를 명확히 맞추기
        x_ft_cut = x_ft[:, :, :modes_x, :modes_y, :modes_z_eff]          # (B, C, mx, my, mz)
        weight_cut = self.weight[:, :, :modes_x, :modes_y, :modes_z_eff] # (C, O, mx, my, mz)

        # einsum: (B, C, mx, my, mz), (C, O, mx, my, mz) → (B, O, mx, my, mz)
        out_ft = torch.zeros(B, self.out_channels, nx, ny, modes_z, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes_x, :modes_y, :modes_z_eff] = torch.einsum(
            "bcmxy,comxy->bomxy", x_ft_cut, weight_cut
        )
        x_ifft = torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=[2,3,4])
        x_local = self.w_local(x)
        return x_ifft + x_local
    
class PSDTransformerNO(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size=8, n_heads=4, n_layers=4, pde_param_dim=1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.pde_param_dim = pde_param_dim
        self.n_layers = n_layers

        self.patch_embed = nn.Linear(in_dim, self.embed_dim)
        self.pde_mlp = nn.Linear(self.pde_param_dim, self.embed_dim)
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(self.embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(self.embed_dim, out_dim)

    def forward(self, x, pde_params=None, grid_shape=None):
        N, in_dim = x.shape
        # 3D grid화
        if grid_shape is not None:
            nx, ny, nz = grid_shape
            try:
                x = x.view(nx, ny, nz, in_dim)
                x = x.view(-1, in_dim)
            except Exception as e:
                print(f"[PSDTransformerNO] grid_shape reshape 실패: {e}, fallback to flat")
                pass

        patches = self.patch_embed(x)  # (N, embed_dim)
        # PDE 파라미터 embedding
        if pde_params is not None:
            if len(pde_params.shape) == 1:
                pde_params = pde_params.unsqueeze(0)
            pde_feat = self.pde_mlp(pde_params)
            if pde_feat.shape[0] == 1:
                pde_feat = pde_feat.expand_as(patches)
            elif pde_feat.shape[0] != patches.shape[0]:
                pde_feat = pde_feat.repeat(patches.shape[0], 1)
            patches = patches + pde_feat
        patches = patches.unsqueeze(0)  # (B=1, N, embed_dim)

        y = patches
        for block in self.transformer_blocks:
            y = block(y)
        y = self.out_proj(y)
        y = y.squeeze(0)  # (N, out_dim)
        return y
    
# --------- Neural Operator Decoder (with UQ) ---------
class ComplexDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.real_drop = nn.Dropout(p)
        self.imag_drop = nn.Dropout(p)
    def forward(self, x):
        return torch.complex(self.real_drop(x.real), self.imag_drop(x.imag))

# =========================================================================================
# 2. PARONet_Decoder 클래스를 아래 코드로 대체
# =========================================================================================
class PARONet_Decoder(nn.Module):
    def __init__(self, coord_dim=3, latent_dim=256, out_dim=8, dropout_p=0.2,
                 region_params=None):
        super().__init__()
        self.coord_layer = nn.Linear(coord_dim, latent_dim)
        self.latent_to_field = ComplexLinear(latent_dim * 2, out_dim)
        self.dropout = ComplexDropout(dropout_p)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.cfloat))
        self.region_params = region_params
        
        # SoftRegionOperator를 새로운 PARONetOperatorSelector로 교체
        self.operator_selector = PARONetOperatorSelector(latent_dim * 2, out_dim, n_ops=4)
        
        # RL 에이전트가 선택한 운영자 설정을 저장할 변수
        self.region_operator_config = None

    def forward(self, coords, latent, for_boundary=False):
        coords = coords.float()
        coord_feat = self.coord_layer(coords)
        latent_rep = latent.repeat_interleave(coords.shape[0] // latent.shape[0], dim=0)
        
        # 입력 피처를 복소수가 아닌 실수로 준비
        combined_real = torch.cat([coord_feat, latent_rep.real], dim=1)  # (N, latent_dim*2)

        # 경계 조건 계산 시에는 가장 간단한 MLP 운영자를 사용
        if for_boundary:
            out = self.operator_selector.operators[0](combined_real) # MLP 직접 호출
        else:
            # ------ 지역별 pde_params 생성 -------
            # 이 부분은 config 파일에서 관리하는 것이 더 유연함
            k_bg = self.region_params.get('k_bg', 730.0)
            k_target = self.region_params.get('k_target', 750.0)
            k_boundary = self.region_params.get('k_boundary', 720.0)
            
            region_pde_map = {
                0: torch.tensor([[k_bg]], device=coords.device),      # 배경
                1: torch.tensor([[k_target]], device=coords.device),  # 타겟
                2: torch.tensor([[k_boundary]], device=coords.device) # 경계
            }
            
            # 임시로 region_mask를 여기서 생성하여 pde_params를 매핑
            temp_region_mask = get_region_mask(coords, **self.region_params)
            
            pde_params_all = torch.zeros(coords.shape[0], 1, device=coords.device)
            for reg_idx, pde_param in region_pde_map.items():
                pde_params_all[temp_region_mask == reg_idx] = pde_param

            # 핵심: operator_selector 호출 시 RL 에이전트의 선택(op_config)을 전달
            out = self.operator_selector(
                x=combined_real,
                coords=coords,
                pde_params=pde_params_all,
                op_config=self.region_operator_config,
                region_params=self.region_params
            )

        # 최종 출력에 bias와 dropout 적용
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

        # Boundary loss 계산
        pred_bc = model.decoder(batch.boundary_coords.float(), latent, for_boundary=True)
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

def build_pyg_data_3d(positions, phases, amplitudes, coords, field, knn=4, device='cpu', xy_range=0.06, z_range=(0.01, 0.07)):
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
    coords_batch = torch.zeros(coords.shape[0], dtype=torch.long, device=device)
    data = Data(x=node_feat, edge_index=edge_index, coords=coords, y=y, coords_batch=coords_batch)

    # boundary mask 계산
    mask_b = mask_boundary_points_3d(coords.cpu().numpy(), xy_range=xy_range, z_range=z_range)
    data.boundary_coords = torch.tensor(coords.cpu().numpy()[mask_b], dtype=torch.float32, device=device)
    data.n_boundary = data.boundary_coords.shape[0]
    arr = field[mask_b, :2] if field.ndim > 1 else field[mask_b]
    if arr.ndim == 1:
        arr = arr[None, :]
    data.boundary_p = torch.tensor(arr, dtype=torch.float32, device=device)
    # ----------- 반드시 추가! -----------
    # batch내 index 0부터 시작
    data.boundary_batch = torch.zeros(data.boundary_coords.shape[0], dtype=torch.long, device=device)
    return data

def custom_collate(data_list):
    batch = Batch.from_data_list(data_list)
    batch.boundary_p_list = [d.boundary_p for d in data_list]
    batch.n_boundary_list = [d.boundary_coords.shape[0] for d in data_list]
    boundary_batches = []
    for i, d in enumerate(data_list):
        n_b = d.boundary_coords.shape[0]
        boundary_batches.append(torch.full((n_b,), i, dtype=torch.long))
    if boundary_batches:
        batch.boundary_batch = torch.cat(boundary_batches).to(batch.boundary_coords.device)
    return batch

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
        mask_b = mask_boundary_points_3d(coords, xy_range, z_range)
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
        arr = field[mask_b, :2]
        if arr.ndim == 1:
            arr = arr[None, :]
        assert arr.ndim == 2 and arr.shape[1] == 2, f"boundary_p shape error: {arr.shape}"
        data.boundary_p = torch.tensor(arr, dtype=torch.float32, device=device)
        data_list.append(data)
    return data_list

def custom_collate(data_list):
    batch = Batch.from_data_list(data_list)
    batch.boundary_p_list = [d.boundary_p for d in data_list]
    batch.n_boundary_list = [d.boundary_coords.shape[0] for d in data_list]
    boundary_batches = []
    for i, d in enumerate(data_list):
        n_b = d.boundary_coords.shape[0]
        boundary_batches.append(torch.full((n_b,), i, dtype=torch.long))
    if boundary_batches:
        batch.boundary_batch = torch.cat(boundary_batches).to(batch.boundary_coords.device)
    return batch

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
    """
    PARONet 모델을 학습시키는 메인 트레이닝 함수.
    데이터 손실, 물리 정보 손실(PDE, BC), 지역별 손실을 모두 사용하여 모델을 최적화하고,
    선택적으로 RL 에이전트를 통해 운영자 선택을 동적으로 조정합니다.

    Args:
        model (nn.Module): 학습할 PARONet 모델.
        train_loader (DataLoader): 학습 데이터 로더.
        device (str): 학습에 사용할 디바이스 ('cuda' or 'cpu').
        y_mean (np.ndarray): 정규화에 사용된 타겟 변수들의 평균.
        y_std (np.ndarray): 정규화에 사용된 타겟 변수들의 표준편차.
        target_coord (list or tuple): 압력을 극대화할 목표 지점의 좌표.
        cfg (dict): 각종 하이퍼파라미터가 담긴 설정 딕셔너리.
        region_params (dict): 지역 마스크 생성에 필요한 파라미터.
        k (float): 헬름홀츠 방정식의 파수(wavenumber).
        epochs (int): 총 학습 에포크 수.
        agent (RegionOpRLAgent, optional): 지역별 운영자를 선택하는 강화학습 에이전트. Defaults to None.
        lr (float, optional): 학습률. Defaults to 1e-5.
        w_pde (float, optional): PDE 손실의 가중치. Defaults to 1.0.
        w_bc (float, optional): 경계 조건 손실의 가중치. Defaults to 1.0.
        w_target (float, optional): 목표 지점 손실의 가중치. Defaults to 1.0.
        logdir (str, optional): 로그 및 모델 저장 경로. Defaults to "./logs".

    Returns:
        list: 에포크별 학습 손실 기록.
    """
    # 1. 최적화기, 스케줄러, 손실 가중치 등 초기 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5, verbose=False)
    early_stop = EarlyStopping(patience=cfg.get('patience', 100), warmup=cfg.get('warmup', 100))
    loss_hist = []

    # 손실 함수 가중치 로드
    w_target_region = cfg.get('w_target_region', 1.0)
    w_boundary_region = cfg.get('w_boundary_region', 1.0)
    w_background_region = cfg.get('w_background_region', 1.0)

    # 2. 전체 학습 에포크 루프
    for epoch in tqdm(range(epochs), desc="Training PARONet"):
        model.train()
        epoch_total_loss = 0

        # 에포크 시작 시, RL 에이전트가 운영자 조합(정책)을 선택
        if agent:
            region_operator_config = agent.select_operators()
            model.decoder.region_operator_config = region_operator_config

        # 3. 배치 단위 학습 루프
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # --- 연산 효율화를 위해 latent 벡터는 배치당 한 번만 계산 ---
            latent = model.encoder(batch.x, batch.edge_index, batch.x[:, :3], batch.batch)

            # --- 손실 계산 ---
            # 3-1. 데이터 충실도 손실 (Data Fidelity Loss: Pressure & Velocity)
            # 모델의 주 예측값(내부 전체 포인트)
            pred_interior = model.decoder(batch.coords[batch.mask_interior], latent)
            y_interior = torch.complex(batch.y[batch.mask_interior, 0], batch.y[batch.mask_interior, 1])
            loss_pressure = complex_mse_loss(pred_interior[:, 0], y_interior)

            # Velocity는 실수부만 비교
            pred_v_real = torch.cat([pred_interior[:, 1:4].real, pred_interior[:, 1:4].imag], dim=1) # 예시, 모델 출력에 맞게 수정
            y_v_real = batch.y[batch.mask_interior, 2:8]
            loss_velocity = F.mse_loss(pred_v_real, y_v_real)
            loss_data = loss_pressure + loss_velocity

            # 3-2. 경계 조건 손실 (Boundary Condition Loss)
            pred_bc = model.decoder(batch.boundary_coords.float(), latent, for_boundary=True)
            true_bc = torch.complex(batch.boundary_p[:, 0], batch.boundary_p[:, 1])
            loss_bc = complex_mse_loss(pred_bc[:, 0], true_bc)

            # 3-3. 물리 정보 손실 (Physics-Informed Loss: PDE Residual)
            coords_int_full = batch.coords[batch.mask_interior].clone().requires_grad_(True)
            pred_pde = model.decoder(coords_int_full, latent)
            p_val = pred_pde[:, 0]

            # 헬름홀츠 방정식 잔차 계산
            grad_p = torch.autograd.grad(p_val.sum(), coords_int_full, create_graph=True)[0]
            lap_p = torch.zeros_like(p_val.real)
            for i in range(3):
                grad_p_i = grad_p[:, i]
                grad2_p_i = torch.autograd.grad(grad_p_i.sum(), coords_int_full, create_graph=True)[0][:, i]
                lap_p = lap_p + grad2_p_i
            
            pde_residual = lap_p + (k**2) * p_val
            loss_pde = complex_mse_loss(pde_residual, torch.zeros_like(pde_residual))

            # 3-4. 지역별 손실 및 목표 지점 손실 (Region-Specific & Target-Point Losses)
            region_mask = get_region_mask(coords_int_full.detach(), **region_params)
            
            # 목표 지점 압력 극대화
            t_coord = torch.tensor(target_coord, dtype=torch.float32, device=device).unsqueeze(0)
            loss_target = target_point_loss(model, t_coord, latent)

            # 지역별 제어 손실 (타겟: 압력 최대화, 경계/배경: 압력 최소화)
            loss_target_region = -pred_pde[region_mask==1].abs().mean() if (region_mask==1).sum() > 0 else torch.tensor(0.0, device=device)
            loss_boundary_region = (pred_pde[region_mask==2].abs()**2).mean() if (region_mask==2).sum() > 0 else torch.tensor(0.0, device=device)
            loss_background_region = (pred_pde[region_mask==0].abs()**2).mean() if (region_mask==0).sum() > 0 else torch.tensor(0.0, device=device)
            
            # --- 4. 최종 손실 결합 및 역전파 ---
            total_loss = (
                loss_data +
                w_pde * loss_pde +
                w_bc * loss_bc +
                w_target * loss_target +
                w_target_region * loss_target_region +
                w_boundary_region * loss_boundary_region +
                w_background_region * loss_background_region
            )
            
            total_loss.backward()
            optimizer.step()
            epoch_total_loss += total_loss.item()
        
        # 5. 에포크 마무리 (로깅, 스케줄링, RL 에이전트 업데이트)
        avg_epoch_loss = epoch_total_loss / len(train_loader)
        loss_hist.append(avg_epoch_loss)
        scheduler.step(avg_epoch_loss)
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch:04d}] Avg Loss: {avg_epoch_loss:.4e} | PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e} | Data: {loss_data.item():.3e}")
            if agent:
                print(f"  └─ RL Op Config: {model.decoder.region_operator_config}")

        # 강화학습 에이전트 업데이트: 이번 에포크의 정책(operator_config)과 결과(각 지역별 손실)를 바탕으로 정책을 개선
        if agent:
            reward_dict = {
                0: -loss_background_region.item(),
                1: -loss_target_region.item(),
                2: -loss_boundary_region.item(),
            }
            agent.update(region_operator_config, reward_dict)
            
        # 조기 종료 조건 확인
        early_stop(avg_epoch_loss)
        if early_stop.stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # 6. 학습 완료 후 로그 저장
    os.makedirs(logdir, exist_ok=True)
    np.save(os.path.join(logdir, "loss_history.npy"), np.array(loss_hist))
    print("Training finished.")
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

def plot_field_gorkov_3d(U_mean, U_std, X, Y, Z, title="Gor'kov U (slice)", z_idx=8, savedir=None):
    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    plt.title("Mean Gor'kov U")
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
    if cfg is None:
        print("[Fatal] config 파싱 실패. 실행 중단.")
        exit(1)
    cfg = parse_config(cfg)
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
    
    # 데이터 타입 변환
    for d in train_data:
        # y는 복소수 형태를 기대하지 않으므로 실수형 float32로 유지
        d.y = torch.tensor(d.y_np, dtype=torch.float32) 
        d.boundary_p = torch.tensor(d.boundary_p, dtype=torch.float32)
    for d in test_data:
        d.y = torch.tensor(d.y_np, dtype=torch.float32)
        d.boundary_p = torch.tensor(d.boundary_p, dtype=torch.float32)
        
    normalization_stats = (y_mean, y_std)
    
    train_loader = DataLoader(
        train_data,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.get('batch_size_test', 1), # 테스트 배치 사이즈 추가
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
        
    # ############################ 에러 수정 부분 ############################
    # 'target_center' 키를 'target_region_center'로 변경하여 함수 정의와 일치시킴
    region_params = {
        'target_region_center': to_tuple_float(cfg.get('target_coord', [0.0, 0.0, 0.04])),
        'target_radius': to_float(cfg.get('target_radius', 0.005)),
        'boundary_tol': to_float(cfg.get('boundary_tol', 1e-5)),
        'xy_range': to_float(cfg.get('xy_range', 0.06)),
        'z_range': to_tuple_float(cfg.get('z_range', (0.01, 0.07))),
    }
    # ######################################################################

    print("region_params", region_params)
    for k, v in region_params.items():
        print(f"{k}: {v} ({type(v)})")

    target_coord = cfg.get('target_coord', [0.0, 0.0, 0.04])
    
    model = AcousticPINO3D_UQ(
        in_dim=6, ffm_scales=cfg['ffm_scales'], ffm_dim=cfg['ffm_dim'], gnn_dim=cfg['gnn_dim'],
        dec_dim=cfg['dec_dim'], out_dim=8, dropout_p=cfg['dropout_p'],
        latent_dim=256, grid_size=cfg['grid_size_train'],
        region_params=region_params
    ).to(device)
    
    # RL 에이전트 초기화 (3개 지역: 배경, 타겟, 경계)
    agent = RegionOpRLAgent(ops=['fft','fno','mlp','psdtrans'], n_regions=3, epsilon=0.2)
    
    # 4. 학습
    if args.train:
        print('Training (Data + Physics-Informed Loss)...')
        loss_hist = train_model(
            model, train_loader, device, y_mean, y_std, target_coord, cfg,
            region_params, # <-- region_params 전달
            k=cfg['k'],
            epochs=cfg['epochs'], lr=cfg['lr'], w_pde=cfg.get('w_pde', 1.0), 
            w_bc=cfg.get('w_bc', 1.0), w_target=cfg.get('w_target', 1.0),
            logdir=cfg.get('logdir','./logs'), agent=agent # <-- agent 전달
        )
        save_model(model, os.path.join(cfg.get('logdir','./logs'), 'best_model.pt'))
        plt.figure()
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
        
        test_sample = next(iter(test_loader))
        test_sample = test_sample.to(device)
        
        mean_pred, std_pred = predict_mc_dropout(model, test_sample, n_samples=cfg['n_mc'], device=device)
        
        coords, X, Y, Z = sample_query_points_3d(grid_size=cfg['grid_size_test'])
        y_mean_t = torch.from_numpy(y_mean).to(device)
        y_std_t = torch.from_numpy(y_std).to(device)

        # 역정규화
        mean_pred_denorm = mean_pred * y_std + y_mean
        std_pred_denorm = std_pred * y_std

        p_real, p_imag = mean_pred_denorm[:,0], mean_pred_denorm[:,1]
        vx_r, vy_r, vz_r = mean_pred_denorm[:,2], mean_pred_denorm[:,3], mean_pred_denorm[:,4]
        vx_i, vy_i, vz_i = mean_pred_denorm[:,5], mean_pred_denorm[:,6], mean_pred_denorm[:,7]
        
        U_mean = gorkov_potential_torch_3d(
            torch.tensor(p_real), torch.tensor(p_imag),
            torch.tensor(vx_r), torch.tensor(vy_r), torch.tensor(vz_r),
            torch.tensor(vx_i), torch.tensor(vy_i), torch.tensor(vz_i)
        ).numpy().reshape(cfg['grid_size_test'],cfg['grid_size_test'],cfg['grid_size_test'])
        
        # 불확실성 시각화 (예: 압력의 표준편차)
        p_std = np.sqrt(std_pred_denorm[:,0]**2 + std_pred_denorm[:,1]**2)
        U_std_reshaped = p_std.reshape(cfg['grid_size_test'],cfg['grid_size_test'],cfg['grid_size_test'])

        plot_field_gorkov_3d(U_mean, U_std_reshaped, X, Y, Z, z_idx=cfg['grid_size_test']//2, savedir=cfg.get('logdir','./logs'))
# --------- CLI 실행 ---------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta-PINO-AFC 논문 전체 실험 파이프라인")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args, unknown = parser.parse_known_args()
    main(args)
