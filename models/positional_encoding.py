import torch
from torch import nn


#https://stackoverflow.com/questions/68477306/positional-encoding-for-time-series-based-data-for-transformer-dnn-models#:~:text=Positional%20encoding%20is%20just%20a,about%20position%20of%20each%20word.
class PositionalEncodingLayer(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_angles(self, positions, indexes):
        d_model_tensor = torch.FloatTensor([[self.d_model]]
                                           ).to(positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def forward(self, input_sequences):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        positions = torch.arange(input_sequences.size(1)).unsqueeze(1).to(
            input_sequences.device
        )  # [seq_len, 1]
        indexes = torch.arange(self.d_model).unsqueeze(0).to(
            input_sequences.device
        )  # [1, d_model]
        angles = self.get_angles(positions, indexes)  # [seq_len, d_model]
        angles[:, 0::2] = torch.sin(
            angles[:, 0::2]
        )  # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(
            angles[:, 1::2]
        )  # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(
            input_sequences.size(0), 1, 1
        )  # [batch_size, seq_len, d_model]
        return position_encoding


class InputEmbeddingAndPositionalEncodingLayer(nn.Module):

    def __init__(self, vocab_size, max_len, d_model, dropout):
        super(InputEmbeddingAndPositionalEncodingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncodingLayer(
            d_model=d_model, max_len=max_len
        )

    def forward(self, sequences):
        """
        :param Tensor[batch_size, seq_len] sequences
        :return Tensor[batch_size, seq_len, d_model]
        """
        token_embedded = self.token_embedding(
            sequences
        )  # [batch_size, seq_len, d_model]
        position_encoded = self.position_encoding(
            sequences
        )  # [batch_size, seq_len, d_model]
        return self.dropout(
            token_embedded
        ) + position_encoded  # [batch_size, seq_len, d_model]
