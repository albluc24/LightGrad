#usefull findings:
## preprocess stuff
preprocess creates a bunch of json files, like: 
phn2id.json  test_dataset.json  train_dataset.json  valid_dataset.json
### phn2id.json  
seems to phonetize with nltk and using this as tokenizer
{"-": 1, "<blank>": 2, "<bos>": 3, "<eos>": 4, "!": 5, "'": 6, "(": 7, ")": 8, ",": 9, ".": 10, ":": 11, ";": 12, "?": 13, " ": 14, "AA": 15, "AA0": 16, "AA1": 17, "AA2": 18, "AE": 19, "AE0": 20, "AE1": 21, "AE2": 22, "AH": 23, "AH0": 24, "AH1": 25, "AH2": 26, "AO": 27, "AO0": 28, "AO1": 29, "AO2": 30, "AW": 31, "AW0": 32, "AW1": 33, "AW2": 34, "AY": 35, "AY0": 36, "AY1": 37, "AY2": 38, "B": 39, "CH": 40, "D": 41, "DH": 42, "EH": 43, "EH0": 44, "EH1": 45, "EH2": 46, "ER": 47, "ER0": 48, "ER1": 49, "ER2": 50, "EY": 51, "EY0": 52, "EY1": 53, "EY2": 54, "F": 55, "G": 56, "HH": 57, "IH": 58, "IH0": 59, "IH1": 60, "IH2": 61, "IY": 62, "IY0": 63, "IY1": 64, "IY2": 65, "JH": 66, "K": 67, "L": 68, "M": 69, "N": 70, "NG": 71, "OW": 72, "OW0": 73, "OW1": 74, "OW2": 75, "OY": 76, "OY0": 77, "OY1": 78, "OY2": 79, "P": 80, "R": 81, "S": 82, "SH": 83, "T": 84, "TH": 85, "UH": 86, "UH0": 87, "UH1": 88, "UH2": 89, "UW": 90, "UW0": 91, "UW1": 92, "UW2": 93, "V": 94, "W": 95, "Y": 96, "Z": 97, "ZH": 98}
### test_dataset.json  
is the dataset template, a dict for every entry.
[{"name": "LJ029-0196", "wav_path": "../LJSpeech-1.1/wavs/LJ029-0196.wav", "text": "and that he would visit San Antonio, Houston, Fort Worth, Dallas, and Austin.", "phonemes": ["AH0", "N", "D", "-", "DH", "AE1", "T", "-", "HH", "IY1", "-", "W", "UH1", "D", "-", "V", "IH1", "Z", "AH0", "T", "-", "S", "AE1", "N", "-", "AE0", "N", "T", "OW1", "N", "IY0", "OW0", ",", "HH", "Y", "UW1", "S", "T", "AH0", "N", ",", "F", "AO1", "R", "T", "-", "W", "ER1", "TH", ",", "D", "AE1", "L", "AH0", "S", ",", "AH0", "N", "D", "-", "AO1", "S", "T", "AH0", "N", "-", "."]}]
### conclusion
might be adapted if it's too hard to eradicate, just create a new tokenizer with '1': 1 and so on leaving eos, bos and pad unchanged.

# LightGrad: Lightweight Diffusion Probabilistic Model for Text-to-speech
Demos are available at: https://thuhcsi.github.io/LightGrad/

## Setup Environment

Install python 3.10.

Then, run:
```bash
git clone --recursive https://github.com/thuhcsi/LightGrad.git
python -m pip install -r requirements.txt
```

## Training
### Preprocess for BZNSYP

Download dataset from [url](https://www.data-baker.com/data/index/TNtts).
Run
```bash
python preprocess.py bznsyp [PATH_TO_DIRECTORY_CONTAINING_DATASET] \
    [PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS] \
    --test_sample_count 200 --valid_sample_count 200
```
This will produce `phn2id.json`, `train_dataset.json`, `test_dataset.json`, `valid_dataset.json` in `[PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS]`.

### Preprocess for LJSpeech

Download dataset from [url](https://keithito.com/LJ-Speech-Dataset/).
Run
```bash
python preprocess.py ljspeech [PATH_TO_DIRECTORY_CONTAINING_DATASET] \
    [PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS] \
    --test_sample_count 200 --valid_sample_count 200
```
This will produce `phn2id.json`, `train_dataset.json`, `test_dataset.json`, `valid_dataset.json` in `[PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS]`.

### Training for BZNSYP

Edit `config/bznsyp_config.yaml`, set `train_datalist_path`, `valid_datalist_path`, `phn2id_path` and `log_dir`.
Run:
```bash
python train.py -c config/bznsyp_config.yaml
```

### Training for LJSpeech

Edit `config/ljspeech_config.yaml`, set `train_datalist_path`, `valid_datalist_path`, `phn2id_path` and `log_dir`.
Run:
```bash
python train.py -c config/ljspeech_config.yaml
```

## Inference

Edit `inference.ipynb`.
Set `HiFiGAN_CONFIG`, `HiFiGAN_ckpt` and `ckpt_path` to corresponding files, respectively.

* Note: `add_blank` in `inference.ipynb` should be the same as that in `LightGrad/dataset.py`.

## References

* Our model is based on [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones).
* [HiFi-GAN](https://github.com/jik876/hifi-gan) is used as vocoder.
