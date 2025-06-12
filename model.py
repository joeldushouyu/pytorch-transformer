import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        #NOTE: features is the same size as hidden_size, which is d_model in the paper
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter (γ), shape: (features,)
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable shift parameter (β), shape: (features,)
    
    
    """
        How layer normalization works:
        1. Do a Z standardization(with eps term to avoid division by zero) on every hidden_vector(token vector) independently regardless of batch_size and sequence_legnth
        2. Then for each hidden_vector[i], apply the learned parameter alpha and bias
        
        Batch Normalization has same step2, but different step 1.
        
        # assume x.shape = [batch=4, seq_len=2, hidden=3]
        
        # layer normalization:
            mean = x[b, s, :].mean()    # shape: scalar
            std  = x[b, s, :].std()     # shape: scalar
            x_norm[b, s, :] = (x[b, s, :] - mean) / std         
        # batch normalization:
            mean = x[:, :, i].mean()    # shape: scalar per feature
            std  = x[:, :, i].std()     # shape: scalar per feature
            x_norm[:, :, i] = (x[:, :, i] - mean) / std
    """
    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch_size, seq_len, 1) # note: keepdim=True keeps the last dimension for broadcasting, so operate on the d_model only
        std = x.std(dim=-1, keepdim=True)    # Shape: (batch_size, seq_len, 1)
        
        #epsion = 1e-6  # Small constant to avoid division by zero, when variance is zero
            
        # Normalize each token vector and apply learned scale/shift
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    # MLP operation apply on every token vector independently
    # The paper uses a two-layer feed-forward network with ReLU activation in between.
    # Thus, allow parallel computation on every token vector independently.
    
    # For performance, it is possible that x[batch_size, seq_len, hidden_size] 
    # can be unroll as [batch_size * seq_len, hidden_size] and then apply the FeedForwardBlock on the last dimension (hidden_size).
    # Then reshape it back to [batch_size, seq_len, hidden_size].    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # in the paper, d_model = 255, dff=2048
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1    # xW1 + b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2   #W2 + b2 

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        
        # apply it on every token vector( 1xd_model) independently
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size. 512 in the paper
        self.vocab_size = vocab_size # Size of the vocabulary. 
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    # basically, this is a pre_computed matrix of shape (seq_len, d_model)
    # While d_model is fixed and must match the model's embedding size, the actual sequence length in inputs to forward() can vary and must be ≤ the seq_len used here.
    
    # In other words, based on different user inputs/training text, the input x in forward function can have seq_len <= seq_len of the PositionEncoding class 
    # we pre computed.
    
    # seq_len is the model's context length — the maximum number of positions we expect in any \
        # input sequenceseq_len is the model's context length — the maximum number of positions we expect in any input sequence
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        # given the formula in the paper, we need to calculate the div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        
        # NOTE: position = index of the token in the sequence
        # apply sine/cosine function to the position vector
        # We combine each position index with different frequency scaling factors (div_term of increasding index) to compute sinusoidal embeddings for each dimension.
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model)) i from 0 to d_model/2-1
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model)), i from 0 to d_model/2-1
        
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # remember X is of shape (batch, seq_len, d_model), where seq_len could <= self.seq_len(the context length of the model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
        # the skip connection
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features) # the layer normaliation
    
        def forward(self, x, sublayer):
            # sublayer is the previous layer
            #return x + self.dropout(sublayer(self.norm(x)))  # a more popular way 
            # but in the paper, it is written as 
            return self.norm(x+sublayer(x))  

class MultiHeadAttentionBlock(nn.Module):
    # Recall for a given ioput of (seq_len, d_model), we want 
    # 1. Make 3 copy of the same input as query, key, value
    # 2. Each query, key, value  multiply with W_q, W_k, W_v[d_model x d_model] respectively to get the query, key, value vectors
    # 3. The splict the result of each multplicaitons into multiple headers of [d_model, d_k(d_model/h)] where h is the number of heads
    
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size/ hidden_size 512 in the paper
        self.h = h # Number of heads  
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq matrix of shape (d_model, d_model) for query
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk 
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo  matrix of shape( h*d_h(same as d_k), d_model), which is the same shape as (d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # d_k size
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len) 
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # do it for all h, in all batch in parallel
        
        
        if mask is not None:
            #NOTE: mask will be broadcast in this case for the Encoding mask
            # recall Encoding mask is  # (1, 1, seq_len) for each batch
            # it will be broadcast to (1, seq_len, seq_len) for each batch, with same rows repeating
            
            # But for output decoding, the mask is a casual mask(upper triangle is all -inf)
            
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9) # almost as -inf  # mask is of shape (batch, 1, seq_len, seq_len) or (batch, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
            
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores  # MM of [seq_len, seq_len] with [seq_len, d_k] to get [seq_len, d_k] for all h in batch in parallel
        # return attention_scores for visualization    


    def forward(self, q, k, v, mask):
        # Casual: Mask for upper traiangle so model don't see future tokens for training
        
        """  Inferance:
        No need if doing greedy decoding, where only the last token is used to predict the next token.
        For paralle inference,
            Parallel inference simulates autoregressive behavior across multiple generated tokens:
            So causal mask is still needed (mask upper triangle) 
            Without it, tokens could "cheat" and attend to the future
        """
        
        
        #FROM GPT: q, k, v hold same value for self attention, but different values for cross attention
        
        
        #do a matrix-multiplication of w_q with q for total of batch_siz in parallel
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  # multiple q with wq
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

    
        # now, split to multipl head 
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)


        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x) #MM of x with W_o in paralle for all batch

class EncoderBlock(nn.Module):
    # Encoude use self_attention
    # Basically on eof the encoder blocks.
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # 2 of residual connection

    def forward(self, x, src_mask):
        # src_mask: mask to input encoder, to hide padding words to prevent it affect inputs
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  # same x for QKV in self-attention
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    # multiple layers of EncoderBlock
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)  # Look like this layer of normalization is additional added, not specified in the paper?

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block # K, V from Encode block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # src_mask: mask from encoder: It is used to hide the input padding that needs to fit the context length
        # tgt_mask: mask for decoder: Used during training to prevent the decoder from attending to future tokens (causal mask) or during inference
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features) # look no need norm for the original paper

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    # Linear layer, map d_model size back to vocabulary 
    # projecting the embedding/hidden_space vectors of size d_model back to the vocabulary size
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)   # projection of d_model to vocab_size for all batch in parallel?
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # source langauge
        self.tgt_embed = tgt_embed # target language
        self.src_pos = src_pos # source positional encoding
        self.tgt_pos = tgt_pos # target positional encoding, should be same as source positional encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # src_seq_lent could be different from tgt_seq_len, but in practice, they are usually the same.
    
    # src_seq_len: determine maximu length of source sequence it will affect the positional encoding matrix size and the model
    # decode_Seq_len: determine maximum length of target sequence it will affect the positional encoding matrix size and the model
    # during training, set both to be the same, max value?
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer