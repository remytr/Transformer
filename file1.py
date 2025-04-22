import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # Calls the constructor of the parent class (nn.Module)
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size # Stores the dimensionality of the input embeddings
        self.heads = heads # Defines the number of attention heads in the multi-head attention mechanism
        # Using multiple heads allows the model to attend to different parts of the input sequence in parallel.

        # Calculates the dimensionality of each individual attention head
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be di by heads"

        # Creates a linear layer for transforming the input into value vectors for each head
        # Takes head_dim as input and outputs a vector of the same size
        # Will not have a bias term.
        self.values = nn.Linear(self.head_dim, self.head_dim, bias= False)
        # This linear layer transforms the input into key vectors for each head.
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # Linear layer transforms input to query vectors
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # Projects combined output back to original embed_size
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Gets batch size N, the number of independent sequences being processed in parallel
        N = query.shape[0]

        # Extract the sequence lengths for the value, key, and query inputs.
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)

        # Generalized tensor contraction. For each head and each batch, this performs a dot product between each query vector
        # and each key vector.
        # Energy score that indicates how well each query matches with each key
        energy = torch.einsum("nqhd, nkhd-->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            # Masks are useful in transformer models for padding and causal masking
            # If a mask is provided, positions in the energy tensor are filled where the mask is 0 with a large -ve number.
            # With softmax function, these values lead to near-zero attention weights.
            energy = energy.masked_fill(mask ==0, float("-1e20")) # If the element of the mask is 0 then shut off
            # Mask will be a triangular matrix

        # Calculates attention weights
        # Scaled down by the sqrt(embed_size). Softmax function applied to normalize the energy scores for each query across all keys.
        # Attention tensor has shape of (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql, nlhd--> nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # For each head and each batch it takes the attention weights and does a weighted sum of the value vectors.
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        # This layer performs a linear transformation to project the output back to original embed_size. Allows the
        # multi-head attention mechanism to produce an output that has the same dimensionality as the input
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # LayerNorm takes an average for every example.
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = edevice
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,


                )
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,embed_size,num_layers,heads,forward_expansion,dropout,device,max_length,):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size,heads,forward_expansion,dropout,device) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x) # Get a prediction of what is next.

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout = 0, device="cuda",max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)