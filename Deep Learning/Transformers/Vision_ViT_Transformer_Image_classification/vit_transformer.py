import torch
import torch.nn as nn
import torch.utils.data
import math
import torch.nn.functional as F


class MultiHeadAttentionPart(nn.Module):

    def __init__(self, num_heads, embed_dim, dropout_rate=0.1):
        super(MultiHeadAttentionPart, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.d_head = self.embed_dim // self.num_heads

        self.qkv_linear = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        self.concat_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, qkv):
        query, key, value = self.qkv_linear(qkv).split(self.embed_dim, dim=-1)  # [batch_size, patch_len+1, embed_dim]

        # [batch_size, patch_len, embed_dim] is the input dim, shoud be in the end the same
        query = query.view(query.shape[0], -1, self.num_heads, self.d_head).permute(0, 2, 1,
                                                                                    3)  # [batch_size, num_heads, patch_len+1, d_head]
        key = key.view(key.shape[0], -1, self.num_heads, self.d_head).permute(0, 2, 3,
                                                                              1)  # [batch_size, num_heads, d_head, patch_len+1]
        value = value.view(value.shape[0], -1, self.num_heads, self.d_head).permute(0, 2, 1,
                                                                                    3)  # [batch_size, num_heads, patch_len+1, d_head]

        qk = torch.matmul(query, key) / math.sqrt(
            self.d_head)  # [batch_size, num_heads, patch_len, d_head] @ [batch_size, num_heads, d_head, patch_len] --> [batch_size, num_heads, patch_len+1, patch_len+1]
        softmax_weights = self.dropout(F.softmax(qk, dim=-1))
        qkv = torch.matmul(softmax_weights,
                           value)  # [batch_size, num_heads, patch_len, d_head] @ [batch_size, num_heads, patch_len+1, patch_len+1] --> [batch_size, num_heads, patch_len+1, d_head]
        qkv = qkv.permute(0, 2, 1, 3).contiguous().view(qkv.shape[0], -1, self.num_heads * self.d_head)
        concat_ouput = self.dropout(self.concat_linear(qkv))  # output dim [batch_size, patch_len+1, embed_dim]
        return concat_ouput


class FeedForwardPart(nn.Module):

    def __init__(self, embed_dim, dropout_rate=0.1, expansion_dim=1024):
        super(FeedForwardPart, self).__init__()

        self.embed_dim = embed_dim
        self.expansion_dim = expansion_dim
        self.dropout_rate = dropout_rate

        self.exp = nn.Sequential(
            nn.Linear(self.embed_dim, self.expansion_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.expansion_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, input_seq):
        out = self.exp(input_seq)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.self_attention_multihead = MultiHeadAttentionPart(self.num_heads, self.embed_dim, self.dropout_rate)
        self.ff_layer = FeedForwardPart(self.embed_dim, self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layernorm = nn.LayerNorm(self.embed_dim)

    def forward(self, embedds):
        attention_embedds = self.dropout(embedds + self.self_attention_multihead(self.layernorm(embedds)))
        dropout_embedds = self.dropout(attention_embedds + self.ff_layer(self.layernorm(attention_embedds)))
        return dropout_embedds


class VitTransformer(nn.Module):

    def __init__(self, image_h_w, img_channels, patch_h_w, embed_dim, num_heads, encoder_layers, num_classes,
                 dropout_rate=0.1):
        super(VitTransformer, self).__init__()

        self.image_dim = image_h_w
        self.img_channels = img_channels
        self.patch_dim = patch_h_w
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        assert self.image_dim % self.patch_dim == 0, 'Image dim must be divisible by the patch dim!'

        self.patch_len = (self.image_dim // self.patch_dim) ** 2
        k_or_s_size = (self.patch_dim, self.patch_dim)
        self.conv_chnl_to_embd = nn.Conv2d(in_channels=self.img_channels, out_channels=self.embed_dim,
                                           kernel_size=(k_or_s_size), stride=k_or_s_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.p_e = nn.Parameter(
            torch.randn(1, self.patch_len + 1, self.embed_dim))  # dims: [1, patch_len +1, embed_dim]
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder_module = nn.ModuleList(
            [TransformerEncoder(self.embed_dim, self.num_heads, self.dropout_rate) for _ in range(self.encoder_layers)])
        self.mlp = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, input_seq):
        embedds = self.conv_chnl_to_embd(input_seq).flatten(2).transpose(1,
                                                                         2)  # dims: [batch_size, patch_len, embed_dim]
        batch_cls_tokens = self.cls_token.repeat(embedds.size(0), 1, 1)  # dims: [batch_size, 1, embed_dim]
        embedds = torch.cat((batch_cls_tokens, embedds),
                            dim=1)  # concat cls,  dims: [batch_size, patch_len +1, embed_dim]
        embedds = embedds + self.p_e[:, :embedds.size(1)]
        embedds = self.dropout(embedds)

        for layer in self.encoder_module:
            embedds = layer(embedds)

        cls_embedds = embedds[:, 0]
        output = self.mlp(cls_embedds)
        output = output.view(-1, self.num_classes)  # dims: [batch_size * 1, num_classes]
        return output
