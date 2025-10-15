import torch
import torch.nn as nn
from torchinfo import summary


class ShakespeareLSTM(nn.Module):
    """
    Recurrent Neural Network (RNN) with LSTM units based on https://arxiv.org/pdf/1812.01097

    :vocab_size: Size of the output vocabulary
    :embed_dim: Dimensionality of the embeddings
    :hidden_dim: Number of features in the hidden state
    :num_layers: Number of recurrent layers
    """

    def __init__(
        self,
        vocab_size: int = 80,
        embed_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM Layer with 2 layers of 256 units each
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h_0, c_0)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        final_hidden_state = out[:, -1, :]
        logits = self.fc(final_hidden_state)
        return logits  # logit shape: (batch_size, vocab_size)


if __name__ == "__main__":
    # Instantiate the model and print its summary
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
    output = model(input_data)

    # Output shape should be (batch_size, vocab_size)
    print(output.shape)  # Output should be: torch.Size([10, 80])
