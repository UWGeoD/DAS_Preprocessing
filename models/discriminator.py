import torch.nn as nn


def _conv_block(in_ch, out_ch, stride, norm):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


def _conv_block_t(in_ch, out_ch, norm):
    """Temporal-only stride-(1,2): doubles temporal RF, height unchanged."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class PatchGAN(nn.Module):
    """
    PatchGAN discriminator (LSGAN variant) with configurable temporal receptive field.

    n_layers=1..3: standard stride-2 blocks (both spatial dims)
      n_layers=1 → ~16 samples RF (~3ms)
      n_layers=2 → ~34 samples RF (~7ms)
      n_layers=3 → ~70 samples RF (~14ms)  [default]
    n_layers=4+:  extra temporal-only stride-(1,2) blocks after depth-3, height stays ~3
      n_layers=4 → ~142 samples RF (~28ms)
      n_layers=5 → ~286 samples RF (~57ms)

    Uses InstanceNorm (not BatchNorm) to handle variable batch sizes.
    """
    def __init__(self, in_channels=1, n_layers=3):
        super().__init__()
        n_std   = min(n_layers, 3)
        n_extra = max(n_layers - 3, 0)

        layers = [*_conv_block(in_channels, 64, stride=2, norm=False)]
        nf = 64
        for _ in range(1, n_std):
            nf_next = min(nf * 2, 512)
            layers += _conv_block(nf, nf_next, stride=2, norm=True)
            nf = nf_next
        for _ in range(n_extra):
            layers += _conv_block_t(nf, nf, norm=True)   # hold channel count, just extend temporal RF
        nf_next = min(nf * 2, 512)
        layers += _conv_block(nf, nf_next, stride=1, norm=True)
        layers += [nn.Conv2d(nf_next, 1, kernel_size=4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
