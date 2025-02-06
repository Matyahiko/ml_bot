import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------
# 1. DropPath (Stochastic Depth)
# -----------------------------
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth): https://arxiv.org/abs/1603.09382
    残差接続をランダムにスキップし、深いネットワークの正則化を強化する手法。
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # [batch_size, 1, 1,...] などブロードキャスト可能な形状でマスクを作成
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0/1マスク
        # ドロップする分をスケーリング補正
        output = x / keep_prob * random_tensor
        return output

# -----------------------
# 2. Patch Embedding 層
# -----------------------
class PatchEmbedding(nn.Module):
    """
    画像をパッチ単位に分割し、Conv2d で埋め込み次元に投影
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # B, C, H, W -> B, embed_dim, H//patch_size, W//patch_size
        x = self.proj(x)
        # フラット化: B, D, H', W' -> B, D, N (N = num_patches)
        x = x.flatten(2)
        # 転置: B, D, N -> B, N, D
        x = x.transpose(1, 2)
        return x

# --------------------------------------
# 3. Transformer Encoder Layer with DropPath
# --------------------------------------
class TransformerEncoderLayer(nn.Module):
    """
    通常の Transformer Encoder Layer に DropPath を組み込んだ例
    drop_attn, drop_mlp, drop_path を個別に指定可能
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 drop_attn=0.1, drop_mlp=0.1, drop_path=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(embed_dim * mlp_ratio)

        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_attn, batch_first=False)

        # DropPath for attention branch
        self.drop_path1 = DropPath(drop_prob=drop_path)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_mlp),
            nn.Linear(self.mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_mlp),
        )
        # DropPath for MLP branch
        self.drop_path2 = DropPath(drop_prob=drop_path)

    def forward(self, x):
        """
        x: [seq_len, batch_size, embed_dim]
        """
        # 1) Self-Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)

        # 2) MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.drop_path2(mlp_out)

        return x

# -------------------------
# 4. Vision Transformer 本体
# -------------------------
class VisionTransformer(nn.Module):
    """
    改良版 Vision Transformer
    - 2D Positional Embedding (任意)
    - DropPath 対応
    - cls_token を使わず、patch 全体を avg pooling で分類する例
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_attn=0.1,
        drop_mlp=0.1,
        drop_path_rate=0.0,
        use_2d_pos=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # 2D Positional Embedding または 1D の単純な Pos Embedding のどちらか
        if use_2d_pos:
            # (1, C, H', W') の形で学習パラメータを持つ
            num_patches_per_side = img_size // patch_size  # H' = W' = num_patches_per_side
            self.pos_embed_2d = nn.Parameter(
                torch.zeros(1, embed_dim, num_patches_per_side, num_patches_per_side)
            )
        else:
            # (1, N, D) の形で学習パラメータを持つ
            self.pos_embed_1d = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # ドロップアウト
        self.pos_drop = nn.Dropout(p=drop_attn)

        # Transformer Encoder
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # 深さに応じて drop_path を増やす (線形増加) 例
            drop_path_i = drop_path_rate * i / (depth - 1) if depth > 1 else 0.0
            block = TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_attn=drop_attn,
                drop_mlp=drop_mlp,
                drop_path=drop_path_i
            )
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)

        # 分類ヘッド
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # パラメータ初期化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Position Embedding
        if hasattr(self, 'pos_embed_2d'):
            nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)
        if hasattr(self, 'pos_embed_1d'):
            nn.init.trunc_normal_(self.pos_embed_1d, std=0.02)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, num_classes]
        """
        B = x.size(0)
        # [B, N, D]
        x = self.patch_embed(x)

        # 2D pos embedding
        if hasattr(self, 'pos_embed_2d'):
            # [B, N, D] -> [B, D, H', W']
            num_patches_per_side = int(self.num_patches ** 0.5)
            x_2d = x.transpose(1, 2).reshape(
                B, self.embed_dim, num_patches_per_side, num_patches_per_side
            )
            # 加算
            x_2d = x_2d + self.pos_embed_2d
            # [B, D, H', W'] -> [B, N, D]
            x = x_2d.reshape(B, self.embed_dim, -1).transpose(1, 2)
        else:
            # 1D pos embedding
            x = x + self.pos_embed_1d

        x = self.pos_drop(x)

        # Transformer に入れるために [N, B, D] へ転置
        x = x.transpose(0, 1)

        # 各ブロックを通過
        for blk in self.blocks:
            x = blk(x)

        # 最終LN
        x = self.norm(x)  # [N, B, D]

        # クラス分類トークンを使わず、すべてのパッチの平均をとる
        # x: [N, B, D], N = num_patches
        x = x.mean(dim=0)  # [B, D]

        # 分類ヘッド
        out = self.head(x)  # [B, num_classes]
        return out