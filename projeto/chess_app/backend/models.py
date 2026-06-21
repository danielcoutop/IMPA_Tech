"""
Arquiteturas das redes neurais de avaliação de xadrez.
Extraídas exatamente do notebook `projeto.ipynb` (células 23 e 30),
para garantir compatibilidade com os pesos salvos em .pt.
"""

import torch
import torch.nn as nn


class ChessEvalCNN(nn.Module):
    """
    Baseline: CNN simples para avaliação de posições de xadrez.

    Entrada: tensor (batch, 15, 8, 8)
    Saída:   tensor (batch,) com a avaliação estimada em centipawns
    """

    def __init__(self, in_channels=15, conv_channels=(32, 64, 64), hidden_dim=128, dropout=0.2):
        super().__init__()

        c1, c2, c3 = conv_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        flattened_dim = c3 * 8 * 8

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.head(x)
        return x.squeeze(-1)


class ResidualBlock(nn.Module):
    """Bloco residual básico usado pela ChessEvalResNet."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ChessEvalResNet(nn.Module):
    """
    ResNet para avaliação de posições de xadrez.

    Entrada: tensor (batch, 15, 8, 8)
    Saída:   tensor (batch,) com a avaliação estimada em centipawns
    """

    def __init__(self, in_channels=15, stem_channels=64, block_channels=(64, 128, 128, 256),
                 hidden_dim=256, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        prev_channels = stem_channels
        for out_channels in block_channels:
            blocks.append(ResidualBlock(prev_channels, out_channels, dropout=dropout * 0.5))
            prev_channels = out_channels
        self.res_blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.gap(x)
        x = self.head(x)
        return x.squeeze(-1)


def load_model(architecture: str, weights_path: str, device: str = "cpu"):
    """
    Instancia a arquitetura pedida e carrega os pesos de um arquivo .pt
    (state_dict, exatamente como salvo no notebook via torch.save(model.state_dict(), ...)).
    """
    if architecture == "cnn":
        model = ChessEvalCNN(in_channels=15)
    elif architecture == "resnet":
        model = ChessEvalResNet(in_channels=15)
    else:
        raise ValueError(f"Arquitetura desconhecida: {architecture}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
