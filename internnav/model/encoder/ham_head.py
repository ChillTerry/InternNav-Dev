import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault("SPATIAL", True)

        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)

        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)

        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)

        self.rand_init = args.setdefault("RAND_INIT", True)

        print("spatial", self.spatial)
        print("S", self.S)
        print("D", self.D)
        print("R", self.R)
        print("train_steps", self.train_steps)
        print("eval_steps", self.eval_steps)
        print("inv_t", self.inv_t)
        print("eta", self.eta)
        print("rand_init", self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer("bases", bases)

        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        coef = self.compute_coef(x, bases, coef)

        x = torch.bmm(bases, coef.transpose(1, 2))

        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        bases = bases.view(B, self.S, D, self.R)

        return x

class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        return coef

class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict()):
        super().__init__()

        self.ham_in = nn.Conv2d(ham_channels, ham_channels, 1)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out_conv = nn.Conv2d(ham_channels, ham_channels, 1)
        self.ham_out_norm = nn.BatchNorm2d(ham_channels)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out_conv(enjoy)
        enjoy = self.ham_out_norm(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham

class LightHamHead(nn.Module):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.

    Args:
        ham_channels (int): input feature_channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.
        out_channels (int): Dimension of the output embedding. Default: 512.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(LightHamHead, self).__init__()
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        feature_channels = kwargs['feature_channels']
        dropout_ratio = kwargs.get('dropout_ratio', 0.1)
        in_index = kwargs.get('in_index', -1)
        input_transform = kwargs.get('input_transform', "multiple_select")
        align_corners = kwargs.get('align_corners', False)

        self._init_inputs(in_channels, in_index, input_transform)
        self.feature_channels = feature_channels
        self.out_channels = out_channels
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index
        self.align_corners = align_corners

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        self.ham_channels = ham_channels

        self.squeeze_conv = nn.Conv2d(sum(self.in_channels), self.ham_channels, 1)
        self.squeeze_norm = nn.BatchNorm2d(self.ham_channels)

        self.hamburger = Hamburger(ham_channels, ham_kwargs)

        self.align_conv = nn.Conv2d(self.ham_channels, self.feature_channels, 1)
        self.align_norm = nn.BatchNorm2d(self.feature_channels)

        self.global_query = nn.Parameter(torch.zeros(1, 1, feature_channels))
        trunc_normal_(self.global_query, std=0.02)
        self.embed_proj = nn.Linear(feature_channels, out_channels) if out_channels != feature_channels else None

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def global_embed(self, feat):
        if self.out_channels is None:
            raise ValueError("out_channels must be set for global_embed")
        if self.dropout is not None:
            feat = self.dropout(feat)
        B, C, H, W = feat.shape
        feat_flat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        query = self.global_query.expand(B, -1, -1)  # (B, 1, C)
        attn = torch.bmm(query, feat_flat.transpose(1, 2))  # (B, 1, H*W)
        attn = attn * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        embedding = torch.bmm(attn, feat_flat).squeeze(1)  # (B, C)
        if self.embed_proj is not None:
            embedding = self.embed_proj(embedding)
        return embedding

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        inputs = [
            F.interpolate(level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze_conv(inputs)
        x = self.squeeze_norm(x)
        x = F.relu(x, inplace=True)

        x = self.hamburger(x)

        output = self.align_conv(x)
        output = self.align_norm(output)
        output = F.relu(output, inplace=True)

        embedding = self.global_embed(output)

        return embedding
