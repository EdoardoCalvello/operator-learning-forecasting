import torch
from torch import nn

# Define the neural network model
class SimpleEncoder(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, domain_dim=1, d_model=32, nhead=8, num_layers=6,
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 use_positional_encoding='continuous',
                 append_position_to_x=False,
                 pos_enc_coeff=2,
                 activation='relu',
                 dropout=0.1, norm_first=False, dim_feedforward=2048,
                 query_seq_len=None,
                 query_coords=None,    # (query_seq_len, domain_dim) default coords from config
                 output_seq_len=None):
        super(SimpleEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.domain_dim = domain_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.use_positional_encoding = use_positional_encoding
        self.append_position_to_x = append_position_to_x
        self.pos_enc_coeff = pos_enc_coeff

        self.set_positional_encoding()

        # Standard PyTorch transformer encoder — uses Flash Attention automatically
        # via scaled_dot_product_attention when running on CUDA with PyTorch >= 2.0.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            dim_feedforward=dim_feedforward,
            batch_first=True)  # (batch_size, seq_len, d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False)

        # Cross-attention decoder: coords_y (or a config-derived default) are projected
        # as queries that attend to the encoder output.
        # query_seq_len = int(prediction_horizon / dt), set at construction time.
        self.query_seq_len = query_seq_len
        self.output_seq_len = output_seq_len
        if query_seq_len is not None:
            self.query_proj = nn.Linear(domain_dim, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                dim_feedforward=dim_feedforward,
                batch_first=True)
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=1)
            # Default query coords from config (query_seq_len, domain_dim); used when
            # coords_y is not available (e.g. inference outside the training loop).
            if query_coords is not None:
                self.register_buffer('query_coords', query_coords)
            else:
                self.query_coords = None
        else:
            self.query_proj = None
            self.decoder = None
            self.query_coords = None

        self.linear_in = nn.Linear(input_dim, d_model)
        self.linear_in_position = nn.Linear(input_dim + domain_dim, d_model)
        self.linear_out = nn.Linear(d_model, output_dim)

    def set_positional_encoding(self):
        # for continuous time positional encoding
        even_inds = torch.arange(0, self.d_model, 2).unsqueeze(0)
        odd_inds = torch.arange(1, self.d_model, 2).unsqueeze(0)
        self.register_buffer('even_inds', even_inds)
        self.register_buffer('odd_inds', odd_inds)

    def positional_encoding(self, x, coords):
        # x: (batch_size, seq_len, input_dim)
        # coords: time coordinates for positional encoding
        if self.use_positional_encoding:
            # Generate continuous positional encoding based on coordinates
            pe = torch.zeros(coords.shape[0], self.d_model).to(coords)
            pe[:, 0::2] = torch.sin(10**self.pos_enc_coeff * coords[:,0] * 10**(-4 * self.even_inds / self.d_model))
            pe[:, 1::2] = torch.cos(10**self.pos_enc_coeff * coords[:,0] * 10**(-4 * self.odd_inds / self.d_model))
            for i in range(1, self.domain_dim):
                pe[:, 0::2] = pe[:, 0::2] * torch.sin(10**self.pos_enc_coeff * coords[:,i] * 10**(-4 * self.even_inds / self.d_model))
                pe[:, 1::2] = pe[:, 1::2] * torch.cos(10**self.pos_enc_coeff * coords[:,i] * 10**(-4 * self.odd_inds / self.d_model))
        else:
            # No positional encoding
            pe = torch.zeros(x.shape).to(x)
        return pe

    def apply_positional_encoding(self, x, coords):
        pe = self.positional_encoding(x, coords)
        x += pe
        return x

    def forward(self, x, y=None, coords_x=None, coords_y=None):

        if self.append_position_to_x:
            append = coords_x.permute(2,0,1).repeat(x.shape[0],1,1)
            x = torch.cat((x, append), dim=2)
            x = self.linear_in_position(x)  # (batch_size, seq_len, input_dim+domain_dim)
        else:
            x = self.linear_in(x)  # (batch_size, seq_len, input_dim)

        if self.use_positional_encoding:
            x = self.apply_positional_encoding(x, coords_x)

        if self.use_transformer:
            memory = self.encoder(x)  # (batch_size, seq_len, d_model)
        else:
            memory = x

        if self.decoder is not None:
            queries = self.query_proj(self.query_coords)  # (query_seq_len, d_model)
            queries = queries.unsqueeze(0).expand(memory.shape[0], -1, -1)  # (batch, query_seq_len, d_model)
            x = self.decoder(queries, memory)  # cross-attend queries to encoder output
        else:
            x = memory

        if self.decoder is None and self.output_seq_len is not None:
            if self.output_seq_len > x.shape[1]:
                raise ValueError(
                    f'Encoder-only output length {self.output_seq_len} exceeds '
                    f'input length {x.shape[1]}')
            x = x[:, -self.output_seq_len:, :]

        x = self.linear_out(x)  # (batch_size, query_seq_len or seq_len, output_dim)

        return x
