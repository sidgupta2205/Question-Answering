import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from layers import Conv1dLinear, FeedForward, HighwayEncoder, CharCNN
from util import masked_softmax, stochastic_depth_layer_dropout, get_available_devices
device, _ = get_available_devices()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Transformer Multihead Self-Attention
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0

        self.ma = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True, device=device)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.ma(x, x, x, key_padding_mask = attn_mask.int())
        return attn_output


class ResidualBlock(nn.Module):
    """
    Residual Block
    """

    def __init__(self, module, hidden_size, residual_dropout_p=0.1):
        super().__init__()
        self.module = module
        self.layer_norm = nn.LayerNorm(hidden_size, device=device)
        self.residual_dropout = ops.StochasticDepth(residual_dropout_p, mode='batch')

    def forward(self, x, mask=None):
        
        input = self.layer_norm(x)
        
        output = self.residual_dropout(self.module(input, mask)) if mask != None else self.residual_dropout(self.module(input))
        
        output = output + x
        return output


class Embedding(nn.Module):
    
    CHAR_LIMIT = 16

    
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob, use_char_cnn=False, use_seq=True):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.num_layers = 2

        
        
        vocab_size, char_emb_dim = char_vectors.size(0), char_vectors.size(1)
        self.char_embed = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)

        
        self.char_conv = CharCNN(char_emb_dim=char_emb_dim,
                                hidden_size=hidden_size,
                                kernel_width=5,
                                drop_prob=drop_prob * 0.5,
                                char_limit=self.CHAR_LIMIT) if use_char_cnn else None
        if use_seq:
            self.word_embed = nn.Sequential(
                nn.Embedding.from_pretrained(word_vectors),
                nn.Dropout(drop_prob)
            )
        else:
            self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        emb_dim = word_vectors.size(1) + (hidden_size if use_char_cnn else self.CHAR_LIMIT * char_emb_dim)

        self.proj = Conv1dLinear(emb_dim, hidden_size, bias=False)
        
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idx, c_idx):
        word_emb = self.word_embed(w_idx)   
        char_emb = self.char_embed(c_idx)   

        if self.char_conv == None:
            char_emb = char_emb.view((*char_emb.shape[:2], -1))
            char_emb = F.dropout(char_emb, self.drop_prob * 0.5, self.training)
        else:
            
            char_emb = self.char_conv(char_emb.permute(0, 3, 1, 2)).permute(0, 2, 1) 

        emb = torch.cat((word_emb, char_emb), dim=2)   
        emb = self.proj(emb)  
        emb = self.hwy(emb)   

        return emb


class DepthWiseSeparableConv1d(nn.Module):
    """
    Depth-wise Separable Convolution
    """

    def __init__(self, hidden_size, kernel_size=7):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=hidden_size,
                                    out_channels=hidden_size,
                                    kernel_size=kernel_size,
                                    groups=hidden_size,
                                    padding=kernel_size // 2,
                                    bias=False,
                                    device=device)
        self.point_conv = nn.Conv1d(in_channels=hidden_size,
                                    out_channels=hidden_size,
                                    kernel_size=1,
                                    bias=True,
                                    device=device)
        nn.init.xavier_uniform_(self.depth_conv.weight)
        nn.init.kaiming_normal_(self.point_conv.weight, nonlinearity='relu')
        nn.init.constant_(self.point_conv.bias, 0.0)

    def forward(self, x):
        depth = self.depth_conv(x.transpose(1, 2))
        point = self.point_conv(depth).transpose(1, 2)

        return F.relu(point)


class EncoderBlock(nn.Module):
    """
    QANet Self-Attention Encoder Block Layer
    """

    def __init__(self, hidden_size, num_heads, dropout, kernel_size, num_conv_layers, base_layer_num, total_num_layers):
        super().__init__()

        self.total_num_layers = total_num_layers
        self.drop_prob = dropout

        
        self.pe = PositionalEncoding(hidden_size, dropout)

        
        self.conv = nn.Sequential(
            *[ResidualBlock(
                DepthWiseSeparableConv1d(hidden_size, kernel_size),
                hidden_size=hidden_size,
                residual_dropout_p=stochastic_depth_layer_dropout(self.drop_prob, base_layer_num + i, self.total_num_layers)) for i in range(num_conv_layers)])

        
        self.multihead_att = ResidualBlock(
            MultiHeadAttention(hidden_size, num_heads, dropout),
            hidden_size=hidden_size,
            residual_dropout_p=stochastic_depth_layer_dropout(self.drop_prob, base_layer_num + num_conv_layers, self.total_num_layers))

        
        self.ff = ResidualBlock(
            FeedForward(hidden_size),
            hidden_size=hidden_size,
            residual_dropout_p=stochastic_depth_layer_dropout(self.drop_prob, base_layer_num + num_conv_layers + 1, self.total_num_layers))

    def forward(self, x, mask=None):
        
        x = self.pe(x)
        
        conv = self.conv(x)
        
        att = self.multihead_att(conv, mask)
        
        output = self.ff(att)
        return output


class StackedEmbeddingEncoderBlock(nn.Module):
    """
    Stacked Encoder Block Layer
    """

    def __init__(self, hidden_size, num_blocks, num_heads=8, dropout=0.1, kernel_size=7, num_conv_layers=4):
        super().__init__()
        total_num_layers = (num_conv_layers + 2) * num_blocks
        self.encoders = nn.ModuleList([EncoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                num_conv_layers=num_conv_layers,
                base_layer_num=num_block + 1,
                total_num_layers=total_num_layers) for num_block in range(num_blocks)])

    def forward(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return x


class QANetOutput(nn.Module):
    
    def __init__(self, hidden_size, drop_prob):
        super().__init__()
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1, bias=False)
        
        self.att_linear_2 = nn.Linear(2 * hidden_size, 1, bias=False)
        
        
        

    def forward(self, emb_1, emb_2, emb_3, mask):
        
        logits_1 = self.att_linear_1(torch.cat((emb_1, emb_2), dim=2))
        logits_2 = self.att_linear_2(torch.cat((emb_1, emb_3), dim=2))
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
