import math
from typing import List, Tuple

from transf_utils import to_fp16,  EncoderLayer

class Transformer:
    def __init__(self, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 2048):
        self.d_model = d_model
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff) 
                             for _ in range(num_layers)]
    
    def positional_encoding(self, seq_len: int) -> List[List[float]]:
        pos_enc = [[0.0 for _ in range(self.d_model)] for _ in range(seq_len)]
        
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                div_term = to_fp16(math.exp(to_fp16(i * -math.log(10000.0) / self.d_model)))
                pos_enc[pos][i] = to_fp16(math.sin(to_fp16(pos * div_term)))
                if i + 1 < self.d_model:
                    pos_enc[pos][i + 1] = to_fp16(math.cos(to_fp16(pos * div_term)))
        
        return pos_enc
    
    def forward(self, x: List[List[float]], mask: List[List[float]] = None) -> List[List[float]]:
        seq_len = len(x)
        
        # Add positional encoding
        pos_enc = self.positional_encoding(seq_len)
        x = [[to_fp16(x[i][j] + pos_enc[i][j]) 
              for j in range(min(self.d_model, len(x[i])))]
             for i in range(seq_len)]
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x, mask)
        
        return x

if __name__ == "__main__":
    # Initialize transformer with smaller dimensions for testing
    transformer = Transformer(
        d_model=64,  # Reduced from 512
        num_heads=4,  # Reduced from 8
        num_layers=2,  # Reduced from 6
        d_ff=256     # Reduced from 2048
    )
    
    # Create smaller sample input (batch_size=1, seq_len=4, d_model=64)
    sample_input = [[to_fp16(0.1) for _ in range(64)] for _ in range(4)]
    
    # Create matching mask
    sample_mask = [[1 for _ in range(4)] for _ in range(4)]
    
    # Forward pass
    output = transformer.forward(sample_input, sample_mask)
    
    print("Output shape:", len(output), "x", len(output[0]))
    # Print first few values of output to verify
    print("First few values:", output[0][:5])