"""Zero-dependency UTF-8 byte-level tokenizer supporting English, Chinese, math, and code."""

from __future__ import annotations

from typing import List, Sequence

import torch


class PulseTokenizer:
    """Compact byte-level tokenizer mapping arbitrary UTF-8 text into token IDs in [0, 259]."""

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    SEP_ID: int = 3
    BYTE_OFFSET: int = 4
    VOCAB_SIZE: int = 260

    def encode(
        self,
        text: str,
        max_len: int | None = None,
        add_special_tokens: bool = True,
    ) -> List[int]:
        raw_bytes = text.encode("utf-8", errors="replace")
        ids = [b + self.BYTE_OFFSET for b in raw_bytes]
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
                if add_special_tokens and max_len >= 2:
                    ids[-1] = self.EOS_ID
            elif len(ids) < max_len:
                ids = ids + [self.PAD_ID] * (max_len - len(ids))
        return ids

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        raw = bytearray()
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int == self.EOS_ID:
                break
            if tid_int >= self.BYTE_OFFSET:
                byte_val = tid_int - self.BYTE_OFFSET
                if 0 <= byte_val <= 255:
                    raw.append(byte_val)
        return raw.decode("utf-8", errors="replace")

    def encode_batch(
        self,
        texts: Sequence[str],
        max_len: int | None = None,
        add_special_tokens: bool = True,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        encoded = [
            self.encode(t, max_len=max_len, add_special_tokens=add_special_tokens) for t in texts
        ]
        target_len = max_len if max_len is not None else max((len(x) for x in encoded), default=1)
        padded = [
            (x[:target_len] + [self.PAD_ID] * max(0, target_len - len(x))) for x in encoded
        ]
        return torch.tensor(padded, dtype=torch.long, device=device)
