import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class Config():
    def __init__(self, vocab_size=21128,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_heads=12,
                    pad_token_id=0,
                    initializer_range=0.02,
                    max_position_embedding=512,
                    type_vocab_size=2,
                    hidden_dropout_porb=0.1,
                    dropout=0.1,
                    intermediate_size=3072,
                    hidden_act='gelu',
                    pooler_type=None):
        self.vocab_size = vocab_size 
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.max_position_embedding = max_position_embedding 
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout
        self.hidden_dropout_prob = hidden_dropout_porb

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers

        self.pooler_type = pooler_type
        self.num_heads = num_heads