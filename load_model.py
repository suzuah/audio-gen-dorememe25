import os
import torch
from data import MelodyDataset
from model import MelodyModel
    
def load_model(ckpt_path, tok_path, voc_path, cfg, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")

    assert isinstance(ckpt, dict) and "model" in ckpt # 가중치 state_dict

    dataset = MelodyDataset(tok_path, voc_path, cfg.block_size, cut_at_eos=True)

    V = len(dataset.vocab) # 모델 출력 차원 V
    PAD_ID = dataset.PAD_ID

    model = MelodyModel(
        vocab_size=V,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_hidden_size=cfg.ffn_hidden_size,
        dropout=cfg.dropout,
        block_size=cfg.block_size,
        pad_id=PAD_ID
        ).to(device)
    
    model.load_state_dict(ckpt["model"], strict=True)
    
    model.eval()

    return model, dataset