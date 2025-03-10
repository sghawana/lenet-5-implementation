import math
from typing import List, Tuple

def to_fp16(x: float) -> float:
    ### Simulate 16-bit floating point precision"""
    return round(x * 2**10) / 2**10

def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    ### Matrix multiplication with fp16 precision
    n, m = len(A), len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    
    for i in range(n):
        for j in range(p):
            sum_val = 0.0
            for k in range(m):
                sum_val = to_fp16(sum_val + to_fp16(A[i][k] * B[k][j]))
            C[i][j] = sum_val
    return C

def softmax(x: List[float]) -> List[float]:
    ### Softmax function with fp16 precision
    max_x = max(x)
    exp_x = [to_fp16(math.exp(to_fp16(xi - max_x))) for xi in x]
    sum_exp_x = sum(exp_x)
    return [to_fp16(xi / sum_exp_x) for xi in exp_x]

def layer_norm(x: List[float], epsilon: float = 1e-5) -> List[float]:
    ### Layer normalization with fp16 precision
    mean = to_fp16(sum(x) / len(x))
    var = to_fp16(sum(to_fp16((xi - mean) ** 2) for xi in x) / len(x))
    return [to_fp16((xi - mean) / to_fp16(math.sqrt(var + epsilon))) for xi in x]

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights
        self.W_q = [[to_fp16(0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[to_fp16(0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[to_fp16(0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[to_fp16(0.1) for _ in range(d_model)] for _ in range(d_model)]

    def split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        ### Split into multiple heads
        batch_size = len(x)
        seq_len = len(x[0]) // self.d_model
        
        # Reshape to [batch_size, seq_len, num_heads, d_k]
        result = []
        for b in range(batch_size):
            seq_result = []
            for s in range(seq_len):
                head_result = []
                for h in range(self.num_heads):
                    head = []
                    for i in range(self.d_k):
                        idx = s * self.d_model + h * self.d_k + i
                        if idx < len(x[b]):
                            head.append(x[b][idx])
                        else:
                            head.append(0.0)  # Padding if needed
                    head_result.append(head)
                seq_result.append(head_result)
            result.append(seq_result)
        return result

    def merge_heads(self, x: List[List[List[List[float]]]]) -> List[List[float]]:
        ### Merge heads back together
        batch_size = len(x)
        seq_len = len(x[0])
        
        result = []
        for b in range(batch_size):
            batch_result = []
            for s in range(seq_len):
                for h in range(self.num_heads):
                    batch_result.extend(x[b][s][h])
            result.append(batch_result)
        return result

    def forward(self, query: List[List[float]], key: List[List[float]], 
                value: List[List[float]], mask: List[List[float]] = None) -> List[List[float]]:
        batch_size = len(query)
        seq_len = len(query[0]) // self.d_model
        
        # Linear projections
        Q = matmul(query, self.W_q)
        K = matmul(key, self.W_k)
        V = matmul(value, self.W_v)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Calculate attention scores
        scores = []
        for b in range(batch_size):
            batch_scores = []
            for s in range(seq_len):
                seq_scores = []
                for h in range(self.num_heads):
                    head_scores = []
                    for j in range(seq_len):
                        score = 0.0
                        for i in range(self.d_k):
                            score = to_fp16(score + to_fp16(Q[b][s][h][i] * K[b][j][h][i]))
                        head_scores.append(to_fp16(score / math.sqrt(self.d_k)))
                    seq_scores.append(head_scores)
                batch_scores.append(seq_scores)
            scores.append(batch_scores)
        
        # Apply mask if provided
        if mask is not None:
            for b in range(batch_size):
                for s in range(seq_len):
                    for h in range(self.num_heads):
                        for j in range(seq_len):
                            if j < len(mask) and s < len(mask[j]) and mask[s][j] == 0:
                                scores[b][s][h][j] = float('-inf')
        
        # Apply softmax to each head's attention scores
        attention_weights = []
        for b in range(batch_size):
            batch_weights = []
            for s in range(seq_len):
                seq_weights = []
                for h in range(self.num_heads):
                    seq_weights.append(softmax(scores[b][s][h]))
                batch_weights.append(seq_weights)
            attention_weights.append(batch_weights)
        
        # Apply attention weights to values
        output = []
        for b in range(batch_size):
            batch_output = []
            for s in range(seq_len):
                seq_output = []
                for h in range(self.num_heads):
                    head_output = [0.0] * self.d_k
                    for i in range(seq_len):
                        for d in range(self.d_k):
                            head_output[d] = to_fp16(head_output[d] + 
                                to_fp16(attention_weights[b][s][h][i] * V[b][i][h][d]))
                    seq_output.append(head_output)
                batch_output.append(seq_output)
            output.append(batch_output)
        
        # Merge heads and apply final linear projection
        output = self.merge_heads(output)
        output = matmul(output, self.W_o)
        return output

class PositionWiseFFN:
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights
        self.W1 = [[to_fp16(0.1) for _ in range(d_ff)] for _ in range(d_model)]
        self.W2 = [[to_fp16(0.1) for _ in range(d_model)] for _ in range(d_ff)]
    
    def relu(self, x: float) -> float:
        return to_fp16(max(0.0, x))
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        hidden = matmul(x, self.W1)
        hidden = [[self.relu(x) for x in row] for row in hidden]
        return matmul(hidden, self.W2)

class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
    
    def forward(self, x: List[List[float]], mask: List[List[float]] = None) -> List[List[float]]:
        # Self attention
        attention_output = self.attention.forward(x, x, x, mask)
        
        # Add & Norm
        x = [[to_fp16(x[i][j] + attention_output[i][j]) 
              for j in range(len(x[i]))]
             for i in range(len(x))]
        x = [layer_norm(row) for row in x]
        
        # Feed forward
        ffn_output = self.ffn.forward(x)
        
        # Add & Norm
        x = [[to_fp16(x[i][j] + ffn_output[i][j]) 
              for j in range(len(x[i]))]
             for i in range(len(x))]
        x = [layer_norm(row) for row in x]
        
        return x