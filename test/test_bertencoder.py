import sys

sys.path.append('../')
from model.bertencoder import BertModel, BertEncoder, BertLayer, BertAttention
from model.embedding import BertEmbedding
from model.config import Config
from utils.log_helpers import logger_init
import torch

if __name__ == '__main__':
    logger_init()

    src = torch.tensor([[1, 3, 5, 7, 9, 2, 3], [2, 4, 6, 8, 10, 0, 0]], dtype=torch.long)
    src = src.transpose(0, 1)  # [src_len, batch_size] 
    print(f"input shape [src_len,batch_size]: ", src.shape)

    token_type_ids = torch.LongTensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]]).transpose(0, 1)
    attention_mask = torch.tensor([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, False, False]])
    
    config = Config()
    # BertEmbedding
    bert_embedding = BertEmbedding(config)
    bert_embedding_result = bert_embedding(src, token_type_ids=token_type_ids)
    print(f"BertEmbedding output shape [src_len, batch_size, hidden_size]: ", bert_embedding_result.shape)

    # BertAttention
    bert_attention = BertAttention(config)
    bert_attention_output = bert_attention(bert_embedding_result, attention_mask)
    print(f"BertAttention output shape [src_len, batch_size, hidden_size]: ", bert_attention_output.shape)

    # BertLayer
    bert_layer = BertLayer(config)
    bert_layer_output = bert_layer(bert_attention_output, attention_mask)
    print(f"BertLayer output shape [src_len, batch_size, hidden_size]: ", bert_layer_output.shape)

    # BertEncoder
    bert_encoder = BertEncoder(config)
    bert_encoder_output = bert_encoder(bert_embedding_result, attention_mask)
    print(bert_encoder_output[0].shape)
    print(len(bert_encoder_output))

    # BertModel
    position_ids = torch.arange(src.size()[0]).expand((1, -1))  # [1,src_len]
    bert_model = BertModel(config)
    bert_output = bert_model(input_ids=src, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
    print(bert_output.shape)