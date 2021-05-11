import argparse
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from dataset import TranslationDataset
from model.net import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_pad(tok, max_len: int, indice: List[int]) -> List[int]:
    diff = max_len - len(indice)
    if diff > 0:
        indice += [tok.vocab["[PAD]"]] * diff
    else:
        indice = indice[:max_len]
    assert len(indice) == max_len
    return indice


def get_src_mask(tok, indice: torch.Tensor) -> torch.Tensor:
    return (indice.data == tok.vocab["[PAD]"]).unsqueeze(-2)


def get_tgt_mask(tok, indice: torch.Tensor) -> torch.Tensor:
    mask = (indice.data != tok.vocab["[PAD]"]).unsqueeze(-2)
    mask = mask & subsequent_mask(indice.shape[-1]).type_as(mask.data)
    return ~mask


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


@torch.no_grad()
def greedy_decode(args, tok, model):
    # load checkpoint
    state_dict = torch.load(args.ckpt_path, map_location=torch.device(DEVICE))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()

    # set input
    source = args.source
    enc_inputs = torch.tensor(
        add_pad(tok, args.max_len, tok.encode(source, add_special_tokens=False)),
        device=torch.device(DEVICE),
    )
    enc_inputs = enc_inputs.unsqueeze(0)
    enc_masks = get_src_mask(tok, enc_inputs)

    encoded = model.encoder(model.src_embed(enc_inputs), enc_masks)
    start_token = tok.vocab["[CLS]"]  # [CLS] means <s> in this implementation
    greedy_dec_inputs = torch.zeros(1, args.max_len).type_as(enc_inputs.data)

    # generate greedy decoder input
    next_token = start_token
    for i in tqdm(range(args.max_len)):
        # make input for decoder
        greedy_dec_inputs[0][i] = next_token
        greedy_dec_masks = get_tgt_mask(tok, greedy_dec_inputs)

        # decode
        out = model.decoder(
            model.tgt_embed(greedy_dec_inputs), encoded, enc_masks, greedy_dec_masks
        )
        prediction = model.generator(out).squeeze(0).max(dim=-1, keepdim=False)[1]
        next_token = prediction.data[i].item()

    # translate using greedy decoder input
    dec_masks = get_tgt_mask(tok, greedy_dec_inputs)
    logit = model(enc_inputs, greedy_dec_inputs, enc_masks, dec_masks)
    pred = logit.squeeze(0).max(dim=-1)[1]
    return pred


if __name__ == "__main__":
    # load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Input sequence to translate")
    # parser.add_argument("--ckpt-path", type=str, help="Checkpoint path to decode")
    parser.add_argument(
        "--ckpt-path",  # TODO:
        type=str,
        help="Checkpoint path to decode",
        default="./checkpoints/version-3/best_model_step_361249_loss_2.5004.pt",
    )

    parser.add_argument("--n-enc-block", type=int, default=6)
    parser.add_argument("--n-dec-block", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--fc-hidden", type=int, default=2048)
    parser.add_argument(
        "--num-head", type=int, default=8, help="Number of self-attention head"
    )
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    assert args.source, "You should enter source text to translate."
    assert args.ckpt_path, "You should enter trained checkpoint path."

    # load checkpoint
    tok = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = Transformer(
        vocab_size=len(tok.vocab),
        num_enc_block=args.n_enc_block,
        num_dec_block=args.n_dec_block,
        num_head=args.num_head,
        hidden=args.hidden,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
    )

    # decode
    translated = greedy_decode(args, tok, model)
    end_idx = int(torch.where(translated == tok.vocab["[SEP]"])[0][0])
    result = tok.decode(translated[:end_idx])
    print(result)
