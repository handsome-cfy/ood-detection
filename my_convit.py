import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from continual.convit import HybridEmbed, PatchEmbed, Block, TaskAttentionBlock, ClassAttention, GPSA
from continual.convit_lpa import LPSA


class SemanticBlock(nn.Module):
    def __init__(self,
                 transformer,
                 dim, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 ):
        super(SemanticBlock, self).__init__()

        self.embed_dim = dim

        # 构造semantic_token
        token_list = []
        for _ in range(transformer.num_classes):
            token_list.append(transformer.cls_token)
        self.semantic_tokens = nn.ParameterList(token_list)

        # 构造注意力模块

        # Convit 中的block
        self.SemanticAttention = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer,
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


class OOD_VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=48, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 local_up_to_layer=10, locality_strength=1., use_pos_embed=True, hyper_lambda=None, num_lpsa=5):
        super().__init__()
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = []
        for i in range(depth):
            if i < num_lpsa:
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        attn_type=LPSA,
                        locality_strength=locality_strength,
                        hyper_lambda=hyper_lambda)
                )
            elif i < local_up_to_layer:
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        attn_type=GPSA,
                        locality_strength=locality_strength)
                )
            else:
                self.blocks.append(
                    TaskAttentionBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        attn_type=ClassAttention)
                )

        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        # Semantic Block
        self.semantic_blocks = \
            SemanticBlock(self,
                          dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate
                          )

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.head.apply(self._init_weights)

        # todo: get the ood tokens and detect whether it is ood data
        # after torch.stack, the channel of the ood token is num_classes and the finnal output is 2
        self.ood_head = nn.Linear(self.embed_dim * num_classes, 2) if num_classes > 0 else nn.Identity()
        self.ood_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u, blk in enumerate(self.blocks):
            if u == self.local_up_to_layer:
            # if u == 0:
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)

        x = self.norm(x)
        return x

    def forward_ood_semantic(self, x):
        return self.semantic_blocks(x)

    def forward(self, x):
        x = self.forward_features(x)
        # ood detect
        ood = self.forward_ood_semantic(x)
        # ood_output = ood
        ood_token = torch.stack(ood, 1)
        # ood_token = torch.permute(ood_token,dims=[1,0,2])
        ood_output = ood_token
        #
        # tmp = x[:, 0]
        # ood_token = torch.flatten(ood_token, start_dim=1)
        # # ood_token = torch.squeeze(ood_token, dim=0)
        # # ood_token = torch.permute(ood_token,dims=[1,0])
        # ood_output = self.ood_head(ood_token)

        # classifier
        x = self.head(x[:, 0])
        return x, ood_output


if __name__ == '__main__':
    model = OOD_VisionTransformer(img_size=24, patch_size=2, num_classes=10, depth=2).cuda()
    print("FinishCreate")
    x = torch.randn((2, 3, 24, 24)).cuda()
    out, ood_output = model(x)

    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
