import json
import numpy as np
import yaml
import torch
import soundfile
import sys, os
import onnxruntime as ort
sys.path.append(os.path.dirname(os.getcwd()))
from training import crossconcat
from LightGrad import LightGrad
from joblib import load
from functools import reduce
def toid(phonemes, phn2id):
    """
    phonemes: phonemes separated by ' '
    phn2id: phn2id dict
    """
    #return [phn2id[x] for x in ['<bos>'] + phonemes + ['<eos>']]
    return [phn2id[x] for x in phonemes]

N_STEP = 4
TEMP = 1.5
STREAMING_CLIP_SIZE=200
config_path = 'config/config.yaml'
with open(config_path) as f: config = yaml.load(f, yaml.SafeLoader)

with open(config['phn2id_path']) as f: ids = json.load(f)
vocab_size = len(ids) + 1
#ckpt_path = 'logs/ckpt/LightGrad_7_6357.pt'
ckpt_path = '/users/luca/vanilla.pt'

print('loading ', ckpt_path)
_, _, state_dict = torch.load(ckpt_path, map_location='cpu')
model = LightGrad.build_model(config, vocab_size)
model.load_state_dict(state_dict)
model.eval()
randominput= torch.randint(low=0, high=vocab_size, size=(1, 500))
randominputlen=torch.tensor(len(randominput), dtype=torch.long).unsqueeze(0)
#torch.onnx.export(model, (randominput, randominputlen, 4), 'logs/model.onnx', input_names=['seq','len','timesteps', 'temp', 'stoc','spk','scale','solver'], output_names=['enc','dec','al'])
#model.forward(randominput, randominputlen, 4)
with open(config['valid_datalist_path']) as f: val=json.load(f)
test=ort.InferenceSession('logs/model.onnx')
inv=load('../training/concat_inv.dat')
sound=[]
for sample in val:
    idx=toid(sample['phonemes'], ids)
    seqlen=torch.tensor(len(idx), dtype=torch.long).unsqueeze(0)
    seq=torch.tensor(idx).unsqueeze(0)
    enc, dec, al=model.forward(seq, seqlen, n_timesteps=4)
    #enc, dec, al=test.run(None, {'seq': seq.numpy(), 'len': seqlen.numpy(), 'timesteps': np.array(4), 'temp': np.array(1.0),'spk':None})
    al=al[0,0,:]
    #al= torch.nn.functional.pad(al, (1, 0))
    al= torch.cumsum(al, dim=0).tolist()
    dec=dec[0].T.detach().numpy()
    #for input in test.get_inputs(): print(f"Input name: {input.name}, shape: {input.shape}, type: {input.type}")
    beginning=0
    #dec=np.load(sample['emb_path'])
    for n,u in enumerate(al):
        u=int(u)
        if n!=0: char='-'.join((sample['phonemes'][n], sample['phonemes'][n-1]))
        else: char='-'.join((sample['phonemes'][n], sample['phonemes'][n]))
        embedding=np.mean(dec[beginning:u+1],axis=0)
        beginning=u
        if np.isnan(embedding).any(): breakpoint()
        cands=inv['k'][char].kneighbors([embedding], return_distance=False)[0]
        cand=inv[char][cands[0]]
        #input(cands[0])
        sound.append(cand)
        #if n==len(sample['phonemes'])-1: breakpoint()
    audio=[i['audio'] for i in sound]
    audio=np.concatenate(audio)
    soundfile.write('result.wav', audio, 32000)
    breakpoint()