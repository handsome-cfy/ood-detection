import torch
# from timm.layers import trunc_normal_
from timm.models.layers import trunc_normal_
from torch import nn

import vit.src.model
from vit.src.model import VisionTransformer
from continual.convit import Block
from continual.convit_lpa import LPSA


class ClassAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = fc(dim, dim, bias=qkv_bias)
        self.k = fc(dim, dim, bias=qkv_bias)
        self.v = fc(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_heads=None, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if mask_heads is not None:
            mask_heads = mask_heads.expand(B, self.num_heads, -1, N)
            attn = attn * mask_heads

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        # return x_cls, attn, v
        return x_cls


class SemanticBlock(nn.Module):
    def __init__(self,
                 transformer: vit.src.model.VisionTransformer,
                 num_classes=1000,
                 emb_dim=768,
                 ):
        super(SemanticBlock, self).__init__()

        self.embed_dim = emb_dim

        # 构造semantic_token
        token_list = []
        for _ in range(num_classes):
            token_list.append(transformer.cls_token)
        self.semantic_tokens = nn.ParameterList(token_list)

        # 构造注意力模块

        # Convit 中的block
        self.SemanticAttention = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                attn_type=ClassAttention
            )
        ])

    def forward(self, x):
        B = x.shape[0]

        tokens = []

        mask_heads = None

        for semantic_token in self.semantic_tokens:
            semantic_token = semantic_token.expand(B, -1, -1)

            # 里面就一个block
            for blk in self.SemanticAttention:
                semantic_token = blk(torch.cat((semantic_token, x), dim=1))

            tokens.append(semantic_token[:, 0])

        return tokens


class DomainBlock(nn.Module):
    def __init__(self,
                 transformer: vit.src.model.VisionTransformer,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 head='both',
                 feat_dim=128,
                 contrastive=True,
                 timm=False,
                 locality_strength=1.,
                 hyper_lambda=None
                 ):
        super(DomainBlock, self).__init__()
        self.emb_dim = emb_dim

        self.domain_token = transformer.cls_token
        self.DomainAttention = nn.ModuleList([
            Block(
                dim=self.emb_dim, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                attn_type=LPSA,
                locality_strength=locality_strength,
                hyper_lambda=hyper_lambda
            )
        ])

    def forward(self, x):
        B = x.shape[0]

        mask_heads = None

        domain_token = self.domain_token.expand(B, -1, -1)
        # 里面就一个block
        for blk in self.DomainAttention:
            domain_token = blk(torch.cat((domain_token, x), dim=1))

        return domain_token


class TestModel(nn.Module):
    def __init__(self,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 head='both',
                 feat_dim=128,
                 contrastive=True,
                 timm=False
                 ):
        super(TestModel, self).__init__()

        self.model_type = "vit"
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            attn_dropout_rate=attn_dropout_rate,
            dropout_rate=dropout_rate,
            head=head,
            feat_dim=feat_dim,
            contrastive=contrastive,
            timm=timm
            # 暂时使用默认参数
        )

        self.SemanticBlock = SemanticBlock(
            self.vit,
            num_classes=num_classes,
            emb_dim=emb_dim,
        )

        self.DomainBlock = DomainBlock(
            self.vit,
            num_classes=num_classes,
            emb_dim=emb_dim,
        )

    def forward(self, x):
        if self.model_type == "vit":
            B = x.shape[0]
            # domain_token = self.DomainBlock(x)
            feature = self.vit(x, return_feature=True)

            return self.SemanticBlock(feature)
        return x


if __name__ == '__main__':
    model = TestModel(num_layers=2, num_classes=2).cuda()
    print("FinishCreate")
    x = torch.randn((2, 3, 256, 256)).cuda()
    out = model(x).cpu()

    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
