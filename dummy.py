import torch

src = torch.load("src.pt")
src_mask = torch.load("src_mask.pt")
src_key_padding_mask = torch.load("src_key_padding_mask.pt")
self_attn = torch.load("self_attn.pt")

src2, enc_align = self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

print(src2)
print(enc_align)
