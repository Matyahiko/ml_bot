import torch
import torch.nn as nn
import torch.nn.functional as F

# BayesianTorch の確率的レイヤー
from bayesian_torch.layers import Conv2dReparameterization, LinearReparameterization

################################################################################
# ダウンサンプリング専用のクラス
################################################################################
class BayesianDownsample(nn.Module):
    """
    1x1 のベイジアンConvと BatchNorm をまとめたクラス。
    forward(x) で (out, kl) を返すようにし、
    内部で出力テンソルだけを BatchNorm に渡すことで
    "tuple が dim を持たない" エラーを回避する。
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 prior_mu=0.0, prior_sigma=0.1):
        super(BayesianDownsample, self).__init__()
        self.conv = Conv2dReparameterization(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
            prior_mean=prior_mu,
            prior_variance=prior_sigma**2
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Conv2dReparameterization は (出力テンソル, KL損失) の2要素タプルを返す
        out, kl_conv = self.conv(x)
        # BatchNorm2d はテンソルを期待しているので out だけ渡す
        out = self.bn(out)
        return out, kl_conv


################################################################################
# ベイジアン版の BasicBlock
################################################################################
class BayesianBasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 prior_mu=0.0, prior_sigma=0.1):
        super(BayesianBasicBlock, self).__init__()
        
        # BayesianTorch の Conv2dReparameterization を使用
        self.conv1 = Conv2dReparameterization(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            prior_mean=prior_mu,
            prior_variance=prior_sigma**2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = Conv2dReparameterization(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            prior_mean=prior_mu,
            prior_variance=prior_sigma**2
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        # conv1
        out, kl_conv1 = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # conv2
        out, kl_conv2 = self.conv2(out)
        out = self.bn2(out)
        
        # ダウンサンプリングが必要なら
        if self.downsample is not None:
            identity, kl_down = self.downsample(x)
        else:
            identity = x
            kl_down = 0.0
        
        # ショートカットを足し合わせる
        out += identity
        out = self.relu(out)

        # このブロック内のKL損失を合計
        kl_total = kl_conv1 + kl_conv2 + kl_down
        return out, kl_total


################################################################################
# make_bayesian_layer
################################################################################
def make_bayesian_layer(in_channels, out_channels, block, blocks, stride=1,
                        prior_mu=0.0, prior_sigma=0.1):
    """
    blocks 数だけ BasicBlock (ベイジアン版) を積み重ねる。
    先頭ブロックでダウンサンプリング (stride=2など) が必要なら実施する。
    """
    downsample = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        # ダウンサンプリング専用クラスを使う
        downsample = BayesianDownsample(
            in_channels,
            out_channels * block.expansion,
            stride=stride,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma
        )
    
    layers = []
    # 先頭ブロック
    layers.append(
        block(in_channels, out_channels, stride, downsample,
              prior_mu=prior_mu, prior_sigma=prior_sigma)
    )
    
    # 残りのブロック
    for _ in range(1, blocks):
        layers.append(
            block(out_channels * block.expansion, out_channels,
                  prior_mu=prior_mu, prior_sigma=prior_sigma)
        )
    
    # nn.Sequential ではなく nn.ModuleList にして
    # forward 内でループを回しながら KL を集計する
    return nn.ModuleList(layers)


################################################################################
# 全体のネットワーク
################################################################################
class ModernBayesianCNN(nn.Module):
    def __init__(self, num_classes=10, prior_mu=0.0, prior_sigma=0.1):
        super(ModernBayesianCNN, self).__init__()
        
        # 1st conv (ベイジアンConv)
        self.conv1 = Conv2dReparameterization(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            prior_mean=prior_mu,
            prior_variance=prior_sigma**2
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet風の各層
        self.layer1 = make_bayesian_layer(
            in_channels=64,
            out_channels=64,
            block=BayesianBasicBlock,
            blocks=2,
            stride=1,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma
        )
        self.layer2 = make_bayesian_layer(
            in_channels=64,
            out_channels=128,
            block=BayesianBasicBlock,
            blocks=2,
            stride=2,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma
        )
        self.layer3 = make_bayesian_layer(
            in_channels=128,
            out_channels=256,
            block=BayesianBasicBlock,
            blocks=2,
            stride=2,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma
        )
        self.layer4 = make_bayesian_layer(
            in_channels=256,
            out_channels=512,
            block=BayesianBasicBlock,
            blocks=2,
            stride=2,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma
        )
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 最後の全結合層もベイジアン化
        self.fc = LinearReparameterization(
            in_features=512 * BayesianBasicBlock.expansion,
            out_features=num_classes,
            bias=True,
            prior_mean=prior_mu,
            prior_variance=prior_sigma**2
        )
        
        # ドロップアウト
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward時に KL損失を合算して返す実装例。
        """
        kl_total = 0.0
        
        # conv1
        out, kl_conv1 = self.conv1(x)  # (tensor, kl)
        kl_total += kl_conv1
        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        # layer1 ~ layer4 (ModuleList)
        for block in self.layer1:
            out, kl_block = block(out)
            kl_total += kl_block

        for block in self.layer2:
            out, kl_block = block(out)
            kl_total += kl_block

        for block in self.layer3:
            out, kl_block = block(out)
            kl_total += kl_block

        for block in self.layer4:
            out, kl_block = block(out)
            kl_total += kl_block
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        
        # 全結合層 (最後もベイジアン)
        out, kl_fc = self.fc(out)
        kl_total += kl_fc
        
        # (出力, KL損失)
        return out, kl_total


################################################################################
# テスト用
################################################################################
if __name__ == "__main__":
    # 動作確認
    model = ModernBayesianCNN(num_classes=2)
    test_input = torch.randn(2, 1, 405, 405)  # バッチサイズ=2
    
    # 出力と KL損失
    out, kl = model(test_input)
    print("出力の形状:", out.shape)  # 例: [2, 2]
    print("KL損失:", kl)
