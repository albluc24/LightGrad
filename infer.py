import json
import yaml

import matplotlib.pyplot as plt
import torch

from LightGrad import LightGrad


def convert_phn_to_id(phonemes, phn2id):
    """
    phonemes: phonemes separated by ' '
    phn2id: phn2id dict
    """
    return [phn2id[x] for x in ['<bos>'] + phonemes.split(' ') + ['<eos>']]


def text2phnid(text, phn2id, language='zh', add_blank=True):
    if language == 'zh':
        from text import G2pZh
        character2phn = G2pZh()
        pinyin, phonemes = character2phn.character2phoneme(text)
        if add_blank:
            phonemes = ' <blank> '.join(phonemes.split(' '))
        return pinyin, phonemes, convert_phn_to_id(phonemes, phn2id)
    elif language == 'en':
        from text import G2pEn
        word2phn = G2pEn()
        phonemes = word2phn(text)
        if add_blank:
            phonemes = ' <blank> '.join(phonemes)
        return phonemes, convert_phn_to_id(phonemes, phn2id)
    else:
        raise ValueError(
            'Language should be zh (for Chinese) or en (for English)!')


def plot_mel(tensors, titles):
    xlim = max([t.shape[1] for t in tensors])
    fig, axs = plt.subplots(nrows=len(tensors),
                            ncols=1,
                            figsize=(12, 9),
                            constrained_layout=True)
    for i in range(len(tensors)):
        im = axs[i].imshow(tensors[i],
                           aspect="auto",
                           origin="lower",
                           interpolation='none')
        plt.colorbar(im, ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlim([0, xlim])
    fig.canvas.draw()
    return plt