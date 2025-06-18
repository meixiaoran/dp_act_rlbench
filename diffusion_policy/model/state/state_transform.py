import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Linear layer to transform output to desired dimension
        self.fc = nn.Linear(hidden_dim * 2, input_dim)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: [batch_size=64, seq_len=10, input_dim=10]
        
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        # BiLSTM forward
        out, _ = self.bilstm(x, (h0, c0))  # out: [64, 10, hidden_dim*2]
        
        # Linear layer to get output dimension
        out = self.fc(out)  # out: [64, 10, 10]
        
        return out

# Example usage
if __name__ == "__main__":
    # Create model
    model = BiLSTMModel()
    
    # Create dummy input
    x = torch.randn(64, 10, 10)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")