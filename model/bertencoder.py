import torch
import torch.nn as nn
import torch.nn.functional as F
from mymultiheadattention import MyMultiheadAttention
from config import Config
from embedding import BertEmbedding

"""
    整个BertEncoder由多个BertLayer堆叠形成；
    而BertLayer又是由BertOutput、BertIntermediate和BertAttention这3个部分组成；
    同时BertAttention是由BertSelfAttention和BertSelfOutput所构成。

"""
'''Bert Attention实现'''
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.multi_head_attention = MyMultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
            params 
                query: [tgt_len, batch_size, hidden_size]
                key: [src_len, batch_size, hidden_size]
                value: [src_len, batch_size, hidden_size]
                key_padding_mask: [batch_size, src_len]

            return
                attn_output: [tgt_len, batch_size, hidden_size]
                attn_output_weights: [batch_size, tgt_len, src_len]
        
        """
        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, key_padding_mask=None):
        attn_output = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=None,
            key_padding_mask=key_padding_mask
        )
        attn_output = self.output(attn_output[0], hidden_states)
        return attn_output


'''Bert Layer实现'''
class BertIntermediate(nn.Module):
    """本质上就是全连接层"""
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        # 获得激活函数
        if isinstance(config.hidden_act, srt):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """
            hidden_states : [scr_len, batch_size, hidden_size]
            return : [src_len, batch_size, intermediate_size]
        
        """
        hidden_states = self.dense(hidden_states)
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        """
            hidden_states: [src_len, batch_size, intermediate_size]
            input_tensor [src_len, batch_size, hidden_size] 
            return [src_len, batch_size, hidden_size]

            其中hidden_states是BERTintermediate模块输出，
            input_tensor是BERTAttention模块输出

        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(input_tensor, hidden_states)
        return hidden_states

class BertLayer(nn.Module):
    """将BertAttention、BertIntermediate、BertOutput组成BertLayer"""

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        """
            params
                hidden_states [src_len, batch_size, hidden_size]
                attention_mask [batch_size, src_len]
            return 
                [src_len, batch_size, hidden_size]
        """
        attention_ouptut = self.bert_attention(hidden_states,
                                                key_padding_mask=attention_mask)
        intermediate_output = self.bert_intermediate(attention_output)
        layer_output = self.bert_output(hidden_states, attention_output)
        return layer_output

'''堆叠多个layer组成Bert Encoder'''
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.bert_layers):
            layer_ouput = layer_module(layer_ouput, attention_mask)
            all_encoder_layers.append(layer_output)
        
        return all_encoder_layers
        
    

class BertPooler(nn.Module):
    '''
        输出到下游任务需要预处理
        处理包括线性层和激活函数

    '''
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        '''
            hidden_states: [src_len, batch_size, hidden_states]
            return [batch_size, hidden_size]

        '''
        if self.config.pooler_type == "first_token_transform":
            token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        elif self.config.pooler_type == "all_token_average":
            token_tensor = torch.mean(hidden_states, dim=0)
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

'''Bert模型主体结构'''
class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.bert_embedding = BertEmbedding(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        """
            params
                input_ids [src_len, batch_size]
                attention_mask [batch_size, src_len]
                token_type_ids [batch_size, src_len]
                position_ids [1, src_len]
            return

        """
        embedding_output = self.bert_embedding(input_ids=input_ids, 
                                                position_ids=position_ids,
                                                token_type_ids=token_type_ids)

        all_encoder_outptus = self.bert_encoder(embedding_output, attention_mask=attention_mask)
        sequence_output = all_encoder_outptus[-1]   #取最后一层
        pooled_output = self.bert_pooler(sequence_output)
        return pooled_output, all_encoder_outptus

def get_activation(activation_string):
    act = activation_string.lower()
    if act == 'linear':
        return None
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError("Unspported activation: %s" % act)

