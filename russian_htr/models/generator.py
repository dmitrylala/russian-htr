import torch
from torch import nn
from torchok.models.backbones.resnet import resnet18

from .transformer import Transformer
from .cnn_decoder import FCNDecoder


class Generator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        nhead: int = 8,
        dim_feedforward: int = 512,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.transformer = Transformer(
            hidden_dim, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout, normalize_before=False, return_intermediate_dec=True
        )

        self.cnn_encoder = nn.Sequential(*([
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)] +
            list(resnet18(pretrained=True).children())[1:-2])
        )

        self.char_embedder = nn.Embedding(vocab_size, hidden_dim)

        self.linear_q = nn.Linear(dim_feedforward, dim_feedforward * 2)

        self.cnn_decoder = FCNDecoder(res_norm = 'in')

    @torch.no_grad()
    def forward_queries(self, style_images, queries):
        self.eval()

        features = self.forward_pre(style_images)
        memory = self.transformer.encoder(features.flatten(2).permute(2, 0, 1))

        fake_images = []
        for i in range(queries.shape[1]):
            for j in range(queries.shape[2]):
                query = queries[:, i, j]
                query_emb = self.char_embedder.weight[query].permute(1, 0, 2)
                tgt = torch.zeros_like(query_emb)
                hs = self.transformer.decoder(tgt, memory, query_pos=query_emb)
                image = self.forward_post(hs)
                fake_images.append(image.detach())
        return fake_images

    def forward(self, style_images, query):
        features = self.forward_pre(style_images)
        hs = self.transformer(features, self.char_embedder.weight, query)
        return self.forward_post(hs)

    def forward_pre(self, style_images):
        bs, c, h, w = style_images.shape
        features = self.cnn_encoder(style_images.view(bs * c, 1, h, w))
        # features = self.cnn_encoder(style_images)
        # print(f"After encoder: {features.shape}")
        # return features
        return features.view(bs, self.hidden_dim, 1, -1)

    def forward_post(self, hs):
        h = hs.transpose(1, 2)[-1]
        h = self.linear_q(h)
        h = h.contiguous()
        h = h.view(h.size(0), h.shape[1] // 2, 4, -1)
        h = h.permute(0, 3, 2, 1)
        h = self.cnn_decoder(h)
        return h
