import torch
import torch.nn as nn
import torch.utils.data
import math
import torch.nn.functional as F


class TransformerEmbeddings(nn.Module):

    def __init__(self, embed_size, embed_dim, seq_len, dropout_rate=0.1):
        super(TransformerEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate

        self.p_e = self.positinal_encoding()
        self.embed = nn.Embedding(self.embed_size, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def positinal_encoding(self):

        def get_pe_positions(pos, idx, embed_dim):
            pos_i = math.sin(pos / (10000 ** ((2 * idx) / embed_dim)))
            pos_i_plus_one = math.cos(pos / (10000 ** ((2 * (idx + 1)) / embed_dim)))
            return pos_i, pos_i_plus_one

        p_e_seq_len = self.seq_len * 2  # *2 cuz of handling cases where seq_len higher for the input
        p_e = torch.zeros(p_e_seq_len, self.embed_dim)
        for position in range(p_e_seq_len):  # for each position
            for i in range(0, self.embed_dim, 2):  # for each dimension
                pos_i, pos_i_plus_one = get_pe_positions(position, i, self.embed_dim)
                p_e[position, i] = pos_i
                p_e[position, i + 1] = pos_i_plus_one
        p_e = p_e.unsqueeze(0)
        return p_e

    def forward(self, input_seq):
        curr_seq_size = input_seq.size(1)
        assert curr_seq_size <= 2 * self.seq_len, "Trained sequence length for Positional Encoding is below current required sequence length. Exiting!"
        embedds = self.embed(input_seq) * math.sqrt(self.embed_dim)
        curr_device = embedds.get_device()
        embedds += self.p_e[:, :curr_seq_size].to(curr_device)
        embedds = self.dropout(embedds)
        return embedds


class MultiHeadAttentionPart(nn.Module):

    def __init__(self, num_heads, embed_dim, dropout_rate=0.1):
        super(MultiHeadAttentionPart, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.d_head = self.embed_dim // self.num_heads

        self.query_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.concat_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, q, k, v, mask):
        query = self.query_linear(q)  # [batch_size, seq_len, embed_dim]
        key = self.key_linear(k)  # [batch_size, seq_len, embed_dim]
        value = self.value_linear(v)  # [batch_size, seq_len, embed_dim]

        # [batch_size, seq_len, embed_dim] is the input dim, shoud be in the end the same
        query = query.view(query.shape[0], -1, self.num_heads, self.d_head).permute(0, 2, 1,
                                                                                    3)  # [batch_size, num_heads, seq_len, d_head]
        key = key.view(key.shape[0], -1, self.num_heads, self.d_head).permute(0, 2, 3,
                                                                              1)  # [batch_size, num_heads, d_head, seq_len]
        value = value.view(value.shape[0], -1, self.num_heads, self.d_head).permute(0, 2, 1,
                                                                                    3)  # [batch_size, num_heads, seq_len, d_head]

        qk = torch.matmul(query, key) / math.sqrt(
            self.d_head)  # [batch_size, num_heads, seq_len, d_head] @ [batch_size, num_heads, d_head, seq_len] --> [batch_size, num_heads, seq_len, seq_len]
        qk_masked = qk.masked_fill(mask == 0, -1e10)
        softmax_weights = self.dropout(F.softmax(qk_masked, dim=-1))
        qkv = torch.matmul(softmax_weights,
                           value)  # [batch_size, num_heads, seq_len, d_head] @ [batch_size, num_heads, seq_len, seq_len] --> [batch_size, num_heads, seq_len, d_head]
        qkv = qkv.permute(0, 2, 1, 3).contiguous().view(qkv.shape[0], -1, self.num_heads * self.d_head)
        concat_ouput = self.concat_linear(qkv)  # output dim [batch_size, seq_len, embed_dim]
        return concat_ouput


class FeedForwardPart(nn.Module):

    def __init__(self, embed_dim, dropout_rate=0.1, expansion_dim=1024):
        super(FeedForwardPart, self).__init__()

        self.embed_dim = embed_dim
        self.expansion_dim = expansion_dim
        self.dropout_rate = dropout_rate

        self.exp = nn.Sequential(
            nn.Linear(self.embed_dim, self.expansion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.expansion_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate)
        )
        # self.exp = nn.Linear(self.embed_dim, self.expansion_dim)
        # self.de_exp = nn.Linear(self.expansion_dim, self.embed_dim)
        # self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_seq):
        # out = self.de_exp(self.dropout(F.relu(self.exp(input_seq))))
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

    def forward(self, embedds, mask):
        attention_embedds = self.dropout(self.self_attention_multihead(embedds, embedds, embedds, mask))
        norm_embedds = self.layernorm(attention_embedds + embedds)
        ff_dropout_embedds = self.dropout(self.ff_layer(norm_embedds))
        encoded_embedds = self.layernorm(ff_dropout_embedds + norm_embedds)
        return encoded_embedds


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.self_attention_multihead = MultiHeadAttentionPart(self.num_heads, self.embed_dim, self.dropout_rate)
        self.global_attention_multihead = MultiHeadAttentionPart(self.num_heads, self.embed_dim, self.dropout_rate)
        self.ff_layer = FeedForwardPart(self.embed_dim, self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layernorm = nn.LayerNorm(self.embed_dim)

    def forward(self, embedds, encoded_embedds, input_mask, target_mask):
        q = self.dropout(self.self_attention_multihead(embedds, embedds, embedds, target_mask))
        q = self.layernorm(q + embedds)
        qkv = self.dropout(self.global_attention_multihead(q, encoded_embedds, encoded_embedds, input_mask))
        normed_qkv = self.layernorm(qkv + q)
        ff_layer_qkv = self.dropout(self.ff_layer(normed_qkv))
        decoded_embedds = self.layernorm(ff_layer_qkv + normed_qkv)
        return decoded_embedds


class Transformer(nn.Module):

    def __init__(self, embed_dim, num_heads, encoder_layers, decoder_layers, vocab_size, seq_len):
        super(Transformer, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.embed_layer = TransformerEmbeddings(self.vocab_size, self.embed_dim, self.seq_len)
        self.encoder_module = nn.ModuleList(
            [TransformerEncoder(self.embed_dim, self.num_heads) for _ in range(self.encoder_layers)])
        self.decoder_module = nn.ModuleList(
            [TransformerDecoder(self.embed_dim, self.num_heads) for _ in range(self.decoder_layers)])
        self.logits = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, input_seq, input_mask, target_seq, target_mask):

        embedds = self.embed_layer(input_seq)
        for layer in self.encoder_module:
            embedds = layer(embedds, input_mask)

        target_embedds = self.embed_layer(target_seq)
        for layer in self.decoder_module:
            target_embedds = layer(target_embedds, embedds, input_mask, target_mask)

        output = self.logits(target_embedds)
        vocab_size = output.size(-1)
        output = output.view(-1, vocab_size)  # dims: [batch_size * seq_len, vocab_size]
        return output


class LossCE(nn.Module):

    def __init__(self, smoothing):
        super(LossCE, self).__init__()

        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=self.smoothing)

    def forward(self, preds, labels, mask):
        labels = labels.contiguous().view(-1)  # dims: [batch_size * seq_len]
        mask = mask.float().view(-1)  # dims: [batch_size * max_words]
        non_sum_loss = self.ce(preds, labels)
        loss = (non_sum_loss * mask).sum() / mask.sum()
        return loss


class AdamWithWarpUp:

    def __init__(self, optimizer, embed_dim, warmup_steps):
        self.optimizer = optimizer
        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps

        self.curr_step = 0
        self.curr_lr = 0

    def step(self):
        self.curr_step += 1
        self.lr = self.embed_dim ** (-0.5) * min(self.curr_step ** (-0.5), self.curr_step * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self.optimizer.step()
