import json
import torch
from torch.utils.data import Dataset

class MelodyDataset(Dataset):
    def __init__(self, tok_path, voc_path, block_size=384, cut_at_eos=True, prefix_len=7):
        # block_size: 한 샘플에서 x의 최대 길이(=모델 입력 길이)
        # cut_at_eos: True일 경우 시퀀스를 EOS에서 잘라냄
        # prefix_len: 보존할 토큰 개수(프리픽스)
        self.block_size = block_size
        self.prefix_len = prefix_len
        take = block_size + 1 # # x = seq[:take-1] y = seq[1:take]

        with open(voc_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        # str → idx    
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        # idx → str
        self.itos = {i: s for i, s in enumerate(self.vocab)}
        self.PAD_ID = self.stoi.get("PAD", 0)
        self.EOS_ID = self.stoi.get("EOS", None)

        # 입력 시퀀스 ids를 EOS에서 잘라냄
        def slice_at_eos(ids):
            if not cut_at_eos or self.EOS_ID is None:
                return ids # 원본
            if self.EOS_ID in ids:
                j = ids.index(self.EOS_ID) # 첫 EOS 위치
                return ids[:j+1] # ~EOS 포함
            return ids
            
        # 길이 == take
        def pad_or_trim(seq):
            # 길이 < take: padding
            if len(seq) <= take:
                need = take - len(seq)
                if need > 0:
                    seq = seq + [self.PAD_ID] * need
                return seq[:take]
            
            # 길이 > take: prefix + tail
            p_end = self.prefix_len
            keep = max(0, take - p_end)
            head = seq[:p_end] # prefix
            tail = seq[-keep:] if keep > 0 else []
            out = (head + tail)[:take]
            if len(out) < take:
                out = out + [self.PAD_ID] * (take - len(out))
            return out

        self.samples = []
        with open(tok_path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                obj = json.loads(ln)
                ids = obj["tokens"]

                ids = slice_at_eos(ids)
                ids = pad_or_trim(ids)

                x = ids[:-1] # 입력 시퀀스
                y = ids[1:] # 타깃 시퀀스

                self.samples.append((
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y