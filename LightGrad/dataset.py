import torch
import torch.utils.data
import json
import math
import re
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 datalist_path,
                 phn2id_path,
                 add_blank=False):
        super().__init__()
        with open(datalist_path) as f:
            self.datalist = json.load(f)
        with open(phn2id_path) as f:
            self.phone_set = json.load(f)

        self.add_blank = add_blank
        self.cache = {}

    def get_vocab_size(self):
        # PAD is also considered
        return len(self.phone_set) + 1

    def load_audio_and_melspectrogram(self, audio_path):
        emb=torch.tensor(np.load(audio_path))
        return emb

    def load_item(self, i):
        #item_name, wav_path, text, phonemes = self.datalist[i]
        item_name = self.datalist[i]['name']
        wav_path = self.datalist[i]['emb_path']
        phonemes = self.datalist[i]['phonemes']
        mel=self.load_audio_and_melspectrogram(wav_path)
        if self.add_blank:
            newp=[]
            for p in phonemes:
                newp+=["<blank>",p]
        ph_idx = [self.phone_set[x] for x in phonemes if x in self.phone_set]
        self.cache[i] = {
            'item_name': item_name,
            'ph': phonemes,
            'mel': mel,
            'ph_idx': ph_idx
        }
        return self.cache[i]

    def __getitem__(self, i):
        if i not in self.cache: return self.load_item(i)
        else: return self.cache[i]

    def process_item(self, item):
        return item

    def __len__(self):
        return len(self.datalist)


def collateFn(batch):
    phs_lengths, sorted_idx = torch.sort(torch.LongTensor(
        [len(x['ph_idx']) for x in batch]),
                                         descending=True)

    mel_lengths = torch.tensor([batch[i]['mel'].shape[0] for i in sorted_idx])
    padded_phs = pad_sequence(
        [torch.tensor(batch[i]['ph_idx']) for i in sorted_idx],
        batch_first=True)

    padded_mels = pad_sequence([batch[i]['mel'] for i in sorted_idx],
                               batch_first=True)
    batch_size, old_t, mel_d = padded_mels.shape
    #txts = [batch[i]['txt'] for i in sorted_idx]
    #wavs = [batch[i]['wav'] for i in sorted_idx]
    item_names = [batch[i]['item_name'] for i in sorted_idx]
    if old_t % 4 != 0:
        new_t = int(math.ceil(old_t / 4) * 4)
        temp = torch.zeros((batch_size, new_t, mel_d))
        temp[:, :old_t] = padded_mels
        padded_mels = temp
    return {
        'x': padded_phs,
        'x_lengths': phs_lengths,
        'y': padded_mels.permute(0, 2, 1),
        'y_lengths': mel_lengths,
        'names': item_names
    }


if __name__ == '__main__':
    import tqdm
    #dataset = Dataset('dataset/bznsyp_processed/train_dataset.json',
    #                         'dataset/bznsyp_processed/phn2id.json', 22050,
    #                         1024, 80, 0, 8000, 256, 1024)
    dataset = Dataset('dataset/ljspeech_processed/train_dataset.json',
                      'dataset/ljspeech_processed/phn2id.json', 22050, 1024,
                      80, 0, 8000, 256, 1024)
    #for i in tqdm.tqdm(range(len(dataset))):
    #    dataset[i]
    data = collateFn([dataset[i] for i in range(2)])
    print(data['x'])
    print(data['x_lengths'])
    print(data['y'].shape)
    print(data['y_lengths'])
    print(data['txts'])
    print(data['names'])