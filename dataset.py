import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64) # one number, id of this token
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]  # the origiinal dataset is a list of dictionaries
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        # split the sentences into tokens base on space (since is being trained to split on space)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    
        # Number of padding for both encoder and decorder inputs to reach sequence_length
        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # No need EOS to decoder at training time

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")  #NOTE: because sequence length need to be bigger than all the dataset sentences

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token, # ADD EOS first, then padding
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # During training phase
        # Decoder input: Used by the decoder at training time
        # Format: [SOS] + target tokens + [PADs] (no EOS)
        # This is shifted right so the model learns to predict the *next* token.
        # Example:  ["[SOS]", "I", "like", "cats", "[PAD]", ...]
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Label: What we want the model to output (next-token prediction target)
        # Format: target tokens + [EOS] + [PADs]
        # Example:  ["I", "like", "cats", "[EOS]", "[PAD]", ...]
        # What we expect the model to predict
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token, # ADD EOS first, then padding
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        
        """
        For training of english-> Italian
        For each batch=1
        encoder_input: The english phrase of encoder_input type that 
            1. [SOS]  [token of the english phrase]  [EOS] [PADs]
        Decode input: The Italian phrase of the decoder_input
            [SOS] [Token of Itanlian phrase] [PAD]
        LABEL: The reference output should by model
            [TOKEN of Italian phrase] [EOS] [PAD]
        
        
        For inference
        encoder_input:
            # Format: [SOS] + source tokens + [EOS] + [PADs]
            # Shape: (seq_len,)
            # Use full encoder input since the encoder is not autoregressive.
            
        decoder_input:
            GREEDY:
                # Start with: [SOS] + [PADs]
                # Then iteratively add predicted tokens:
                # [SOS, "I"], [SOS, "I", "like"], etc. 
            PARALLEL: #training-like batch decoding
                # You may pre-allocate: [SOS] + predicted tokens so far + [PADs]
                # The only difference is need of casual maps, because adding multiple tokens at one decode stage
                
            EX: If want to decode 5 tokens
            
                #GREEDY:
                    step 1: [SOS, UNK, EOS, PAD... until context length]
                    step 2: [SOS, T0,  UNK, EOS, PAD... until context length]                 
                    step 3: [SOS, T0,  T1,  UNK, EOS, PAD... until context length]
                    step 4: [SOS, T0,  T1,  T2,  UNK, EOS, PAD... until context length]
                    step 5: [SOS, T0,  T1, T2,   T3,  UNK, EOS, PAD... until context length]
                # PARALLEL
                    step1 :[SOS, UNK, UNK, UNK, UNK, UNK, PAD until context lenght]
                    
                    with casual mask of
                    
                    [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #SOS
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], #T0
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], #T1
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], #T2
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], #T3
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], #T4, see code below 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],# padding rows masked out entirely
                    ...
                    ]                    
        """

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # length of (seq_len)
            "decoder_input": decoder_input,  # length of (seq_len)
            
            # Encoder mask: Masks out padding in the encoder input.
            # Shape: (1, 1, seq_len)
            # Used in self-attention to ignore padding tokens.
            # This mask will be broadcast to shape (batch, num_heads, seq_len, seq_len).            
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),   # During training, mask all [PAD] tokens for encoder  # (1, 1, seq_len)
                                                                                                        # will be broadcast to (1, seq_len, seq_len) during actual attention(rows repeating)
                                                                                                        

            # Decoder mask: Combination of
            # - Padding mask: to ignore [PAD] tokens in the decoder input
            # - Causal mask: to prevent attention to future tokens (autoregressive constraint)
            # Final shape: (1, seq_len, seq_len)
            # Used in the self-attention inside the decoder.
                                                                                                                    
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            # creates the casual map first, & with decoder_inpu mask due to [PAD] in decode
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0