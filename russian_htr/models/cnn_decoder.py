from torch import nn

from .blocks import ResBlocks, Conv2dBlock


class FCNDecoder(nn.Module):
    def __init__(
		self,
		ups=3,
		n_res=2,
		dim=512,
		out_dim=1,
		res_norm='adain',
		activ='relu',
		pad_type='reflect'
	):
        super().__init__()

        layers = []
        layers += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for _ in range(ups):
            layers += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        layers += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
