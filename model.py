import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """
    시퀀스 내 각 위치마다 벡터 부여
    (transformer는 위치 정보를 직접 처리하지 못하므로 위치 임베딩 필요)
    """
    def __init__(self, max_len: int, hidden_size: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, hidden_size) # pos idx(int) → vector

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        # B: batch size
        # L: sequence length
        # D: hidden dimendion
        B, L, D = x.shape
        idx = torch.arange(L, device=x.device) # [L], 0..L-1 위치 인덱스 생성
        pos_emb = self.pos(idx) # [L, D] 각 위치의 임베딩 벡터 조회
        pos_emb = pos_emb.unsqueeze(0) # [1, L, D]
        return x + pos_emb # [B, L, D] 브로드캐스팅
    

class MelodyModel(nn.Module):
    """
    Decoder-only
    input: token id sequence
    token embedding + position embedding
    causal mask 미래 정보 가림
    output: 다음 token에 대한 확률 분포(logits)
    """
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 num_heads,
                 num_layers,
                 ffn_hidden_size,
                 dropout,
                 block_size,
                 pad_id=None):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id

        # token embedding
        # tok ids(int) → embedding vector
        self.tok = nn.Embedding(vocab_size, hidden_size,
                                padding_idx=pad_id if pad_id is not None else None) # gradient 누적 X

        # position embedding
        self.pos = LearnedPositionalEncoding(block_size+100, hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            batch_first=True # [B, L, D]
        )

        # num_layers 쌓아 전체 블록 구성
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(hidden_size) # 정규화
        self.head = nn.Linear(hidden_size, vocab_size, bias=False) # (D → V) logits
        self.head.weight = self.tok.weight # weight tying: 임베딩과 출력 가중치 공유

    # causal mask
    def _future_mask(self, L: int, device):
        # 미래 토큰 위치(대각선 위쪽) -inf, 나머지 0
        # i 현재 토큰 j 참조 토큰 → (i < j) True
        return torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x:torch.Tensor, pad_id:int=None, attn_override=None):
        if pad_id is None:
            pad_id = self.pad_id

        # x: [B, L] 정수 id 토큰
        B, L = x.shape
        
        h = self.tok(x) # [B, L, D]
        h = self.pos(h) # [B, L, D]

        # PAD 위치 mask
        key_padding_mask = (x == pad_id) if pad_id is not None else None

        # attn_mask
        if attn_override is not None:
            attn_mask = attn_override
        else:
            attn_mask = self._future_mask(L, x.device) # [L, L]

        y = self.enc(
            h,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask
            ) # [B, L, D]
        logits = self.head(self.ln(y)) # [B, L, V]
        return logits