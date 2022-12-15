import functools

import torch
import torch.nn as nn
from .gan_layers import SNConv2d, SNLinear, DBlock, Attention
from .vocab import VOCABULARY

# from .networks import init_weights

from .blocks import Conv2dBlock, ResBlocks

class Decoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for _ in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        y =  self.model(x)

        return y


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', input_nc=3):
    arch = {}
    arch[256] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[63] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [input_nc] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    arch[129] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[33] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    arch[31] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    arch[16] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}

    arch[17] = {'in_channels': [input_nc] + [ch * item for item in [1, 4]],
                 'out_channels': [item * ch for item in [1, 4, 8]],
                 'downsample': [True] * 3,
                 'resolution': [16, 8, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}

    arch[20] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}
    return arch


class Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_wide=True, resolution=16,
                 D_attn='64', n_classes=len(VOCABULARY),
                 num_D_SVs=1, num_D_SV_itrs=1,
                 sn_eps=1e-8, output_dim=1,
                 D_init='N02', skip_init=False, gpu_ids=[0], input_nc=1, **kwargs):

        super().__init__()

        # gpu_ids
        self.gpu_ids = gpu_ids

        # Activation
        self.activation = nn.ReLU(inplace=True)  # was False

        # Architecture
        self.arch = D_arch(D_ch, D_attn, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        conv = functools.partial(SNConv2d, kernel_size=3, padding=1,
                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                            eps=sn_eps)

        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=conv,
                                           wide=D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                # print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                                     conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        linear = functools.partial(SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=sn_eps)
        self.linear = linear(self.arch['out_channels'][-1], output_dim)

        # Embedding for projection discrimination
        embedding = functools.partial(SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=sn_eps)
        self.embed = embedding(n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        # if not skip_init:
        #     self = init_weights(self, D_init)

    def forward(self, x, y=None):
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                x = block(x)

        # Apply global sum pooling as in SN-GAN
        # print(x.shape)
        x = torch.sum(self.activation(x), [2, 3])

        # Get initial class-unconditional output
        out = self.linear(x)

        # Get projection of final featureset onto class vectors and add to evidence
        if y is not None:
            out = out + torch.sum(self.embed(y) * x, 1, keepdim=True)
        return out

    def return_features(self, x, y=None):
        block_output = []
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                x = block(x)
                block_output.append(x)

        # Apply global sum pooling as in SN-GAN
        # h = torch.sum(self.activation(h), [2, 3])
        return block_output


class WriterDiscriminator(nn.Module):
    # output_dim=NUM_WRITERS
    def __init__(self, output_dim, D_ch=64, D_wide=True, resolution=16,
                 D_attn='64', n_classes=len(VOCABULARY),
                 num_D_SVs=1, num_D_SV_itrs=1,
                 sn_eps=1e-8,
                 D_init='N02', skip_init=False, gpu_ids=[0], input_nc=1, **kwargs):
        super().__init__()

        # gpu_ids
        self.gpu_ids = gpu_ids
        self.output_dim = output_dim

        # Activation
        self.activation = nn.ReLU(inplace=True)

        # Architecture
        self.arch = D_arch(D_ch, D_attn, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        conv = functools.partial(SNConv2d, kernel_size=3, padding=1,
                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                            eps=sn_eps)
        linear = functools.partial(SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=sn_eps)

        embedding = functools.partial(SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=sn_eps)

        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=conv,
                                           wide=D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                # print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                                     conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = linear(self.arch['out_channels'][-1], output_dim)

        # Embedding for projection discrimination
        self.embed = embedding(n_classes, self.arch['out_channels'][-1])
        self.cross_entropy = nn.CrossEntropyLoss()

        # Initialize weights
        # if not skip_init:
        #     self = init_weights(self, D_init)

    def forward(self, x, y=None):
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                x = block(x)

        # Apply global sum pooling as in SN-GAN
        # print(f"BEFORE GLOBAL: {x.shape}")
        x = torch.sum(self.activation(x), [2, 3])
        # print(f"AFTER GLOBAL: {x.shape}")

        # Get initial class-unconditional output
        out = self.linear(x)
        # print(f"out: {out.shape}, {y.shape}")
        # print(f"{torch.max(out)}")

        # Get projection of final featureset onto class vectors and add to evidence
        # TODO: mb uncomment?
        # if y is not None:
        #     print(y.shape, self.embed.weight.shape)
        #     out = out + torch.sum(self.embed(y) * x, 1, keepdim=True)
        loss = self.cross_entropy(out, y.long())
        return loss

    def return_features(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        block_output = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
                block_output.append(h)
        # Apply global sum pooling as in SN-GAN
        # h = torch.sum(self.activation(h), [2, 3])
        return block_output
