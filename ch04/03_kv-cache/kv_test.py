import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        seq_len = x.shape[0]
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算注意力分数
        attn_scores = queries @ keys.T
        
        # 创建因果掩码（下三角矩阵）
        # 防止当前位置看到未来位置的信息
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        print(f"\nCausal mask shape: {causal_mask.shape}")
        print(f"Causal mask:\n{causal_mask}")
        
        # 应用因果掩码：将上三角部分设为负无穷
        masked_attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        print(f"\nMasked attention scores:\n{masked_attn_scores}")
        
        # 计算注意力权重（缩放点积注意力）
        attn_weights = torch.softmax(masked_attn_scores / keys.shape[-1]**0.5, dim=-1)
        print(f"Causal attention weights:\n{attn_weights}")

        context_vec = attn_weights @ values
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x, layer_name=""):
        print(f"\n=== {layer_name} ===")
        print(f"Input shape: {x.shape}")
        print(f"Input:\n{x}")
        
        # 自注意力 + 残差连接 + 层归一化
        attn_out = self.attention(x)
        print(f"\nAttention output shape: {attn_out.shape}")
        print(f"Attention output:\n{attn_out}")
        
        x_after_attn = self.norm1(x + attn_out)
        print(f"\nAfter attention + residual + norm1:\n{x_after_attn}")
        
        # 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn(x_after_attn)
        print(f"\nFFN output:\n{ffn_out}")
        
        x_final = self.norm2(x_after_attn + ffn_out)
        print(f"\nAfter FFN + residual + norm2 (layer output):\n{x_final}")
        
        return x_final

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        print("=== TRANSFORMER FORWARD PASS ===")
        print(f"Input token ids: {x}")
        
        # Embedding
        x = self.embedding(x)
        print(f"\nAfter embedding - shape: {x.shape}")
        print(f"Embeddings:\n{x}")
        
        # 通过 transformer 层
        for i, layer in enumerate(self.layers):
            x = layer(x, f"Layer {i+1}")
        
        # 输出投影用于下一个token预测
        logits = self.output_projection(x)
        print(f"\n=== FINAL OUTPUT ===")
        print(f"Output logits shape: {logits.shape}")
        print(f"Output logits:\n{logits}")
        
        # 获取下一个token的预测（最后一个位置的logits）
        next_token_logits = logits[-1]  # 最后位置的logits
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_pred = torch.argmax(next_token_logits)
        
        print(f"\nNext token logits: {next_token_logits}")
        print(f"Next token probabilities: {next_token_probs}")
        print(f"Predicted next token ID: {next_token_pred.item()}")
        
        return logits, next_token_pred

# 设置参数
torch.manual_seed(42)
vocab_size = 10  # 简单词汇表大小
d_model = 8      # 嵌入维度
d_ff = 16        # FFN隐藏层维度

# 创建模型
model = SimpleTransformer(vocab_size, d_model, d_ff, num_layers=2)

print("="*80)
print("STEP 1: Generate 5th token from first 4 tokens")
print("="*80)

# 创建输入：4个token
input_tokens_4 = torch.tensor([1, 3, 5, 7])  # 示例token序列
print(f"Input sequence (4 tokens): {input_tokens_4.tolist()}")

# 前向传播
with torch.no_grad():
    logits_4, next_token_5 = model(input_tokens_4)

print(f"\n{'='*50}")
print(f"SUMMARY STEP 1:")
print(f"Input tokens: {input_tokens_4.tolist()}")
print(f"Predicted 5th token: {next_token_5.item()}")
print(f"{'='*50}")

print("\n" + "="*80)
print("STEP 2: Generate 6th token from first 5 tokens (including the predicted 5th)")
print("="*80)

# 现在用前5个token（包括刚预测的第5个）来预测第6个token
input_tokens_5 = torch.cat([input_tokens_4, next_token_5.unsqueeze(0)])
print(f"Input sequence (5 tokens): {input_tokens_5.tolist()}")

# 前向传播
with torch.no_grad():
    logits_5, next_token_6 = model(input_tokens_5)

print(f"\n{'='*50}")
print(f"SUMMARY STEP 2:")
print(f"Input tokens: {input_tokens_5.tolist()}")
print(f"Predicted 6th token: {next_token_6.item()}")
print(f"{'='*50}")

print(f"\n{'='*80}")
print(f"FINAL SUMMARY:")
print(f"Original 4 tokens: {input_tokens_4.tolist()}")
print(f"Generated 5th token: {next_token_5.item()}")
print(f"Generated 6th token: {next_token_6.item()}")
print(f"Complete sequence: {input_tokens_5.tolist()} -> {next_token_6.item()}")
print(f"{'='*80}")

print(f"\n{'='*80}")
print("KEY DIFFERENCE: CAUSAL vs NON-CAUSAL ATTENTION")
print(f"{'='*80}")
print("因果自注意力确保每个位置只能关注它之前（包括当前）的位置")
print("这防止了模型在训练时'偷看'未来的token")
print("注意观察attention weights矩阵中的0值（被掩码的位置）")
print(f"{'='*80}")