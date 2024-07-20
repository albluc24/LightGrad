import json
import yaml
import torch

from LightGrad import LightGrad


def toid(phonemes, phn2id):
    """
    phonemes: phonemes separated by ' '
    phn2id: phn2id dict
    """
    #return [phn2id[x] for x in ['<bos>'] + phonemes.split(' ') + ['<eos>']]
    return [phn2id[x] for x in ['<bos>'] + phonemes + ['<eos>']]

N_STEP = 4
TEMP = 1.5
STREAMING_CLIP_SIZE=200
config_path = 'config/config.yaml'
with open(config_path) as f: config = yaml.load(f, yaml.SafeLoader)

with open(config['phn2id_path']) as f: ids = json.load(f)
vocab_size = len(ids) + 1
#ckpt_path = 'logs/ckpt/LightGrad_7_6357.pt'
ckpt_path = '/users/luca/gradquadro.pt'

print('loading ', ckpt_path)
_, _, state_dict = torch.load(ckpt_path, map_location='cpu')
model = LightGrad.build_model(config, vocab_size)
model.load_state_dict(state_dict)
model.eval()
with open(config['valid_datalist_path']) as f: val=json.load(f)
for sample in val:
    idx=toid(sample['phonemes'], ids)
    seqlen=torch.tensor(len(idx), dtype=torch.long).unsqueeze(0)
    seq=torch.tensor(idx).unsqueeze(0)
    enc, dec, al=model.forward(seq, seqlen, n_timesteps=4)
    al=al[0,0,:]
    breakpoint()