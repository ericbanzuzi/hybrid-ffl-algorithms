import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ShakespeareLSTM(nn.Module):
        """
        Recurrent Neural Network (RNN) with LSTM units based on https://arxiv.org/pdf/1812.01097

        :vocab_size: Size of the output vocabulary
        :embed_dim: Dimensionality of the embeddings
        :hidden_dim: Number of features in the hidden state
        :num_layers: Number of recurrent layers
        """
        def __init__(self, vocab_size: int = 80, embed_dim: int = 8, hidden_dim: int = 256, num_layers: int = 2):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.embed_dim = embed_dim

            self.embedding = nn.Embedding(vocab_size, embed_dim)
            # LSTM Layer with 2 layers of 256 units each
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, x):
            embed = self.embedding(x)
            out, hidden = self.lstm(embed)
            final_hidden_state = out[:, -1, :]  # Shape: (batch_size, hidden_size)
            logits = self.fc(final_hidden_state)
            return logits, hidden  # logit shape: (batch_size, 1, vocab_size)
        

class ShakespeareLSTM2(nn.Module):
    """
    Recurrent Neural Network (RNN) with LSTM units.

    Attributes:
    - no_of_output_symbols (int): Size of the output vocabulary.
    - embedding_size (int): Dimensionality of the embeddings.
    - hidden_size (int): Number of features in the hidden state.
    - num_layers (int): Number of recurrent layers.
    - use_GRU (bool): If True, use GRU; otherwise, use LSTM.
    - dropout (float): Dropout probability.
    - device (torch.device): Device for the model ('cpu', 'mps' or 'cuda').
    """
    def __init__(self, vocab_size, embedding_dim=8, hidden_size=256, num_layers=2, seq_length=80):
        super(ShakespeareLSTM, self).__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM Layer with 2 layers of 256 units each
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h_0, c_0)

    def forward(self, x, hidden):
        embedded = self.embedding(x)  # x shape: (batch_size, seq_length)
        
        # LSTM outputs: output shape (batch_size, seq_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(embedded, hidden)
        final_hidden_state = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Final prediction (scores for each vocabulary item)
        logits = self.fc(final_hidden_state)  # Shape: (batch_size, vocab_size)
        return logits, (h_n, c_n)
    

if __name__ == '__main__':
    from torchinfo import summary    
    
    vocab_size = 80  # For Shakespeare dataset with 80 characters in vocabulary
    model = ShakespeareLSTM(vocab_size=vocab_size)
    summary(model, input_size=(1, 80), dtypes=[torch.long])

    print()
    # TEST:
    # Create a random input tensor (batch_size, sequence_length)
    batch_size = 10  # Number of devices per round
    sequence_length = 80
    input_data = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Initialize hidden states
    # hidden = model.init_hidden(batch_size)

    # Pass the data through the model
    output, _ = model(input_data)

    # Output shape should be (batch_size, vocab_size)
    print(output.shape)  # Output should be: torch.Size([10, 80])
