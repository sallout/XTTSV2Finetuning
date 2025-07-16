# XTTSv2 Arabic Dialect Fine-Tuning (Iraqi Example)

This guide explains how to fine-tune [XTTSv2](https://github.com/coqui-ai/TTS) for high-quality Arabic dialect synthesis (e.g., Iraqi), including environment setup, dataset formatting, and training.

---

## Table of Contents

- [1. Clone the Repository](#1-clone-the-repository)
- [2. Install Dependencies](#2-install-dependencies)
- [3. Download XTTSv2 Base Files](#3-download-xttsv2-base-files)
- [4. Prepare Your Dataset](#4-prepare-your-dataset)
- [5. Move Dataset to Pretrained Folder](#5-move-dataset-to-pretrained-folder)
- [6. Start Training](#6-start-training)
- [7. Troubleshooting](#7-troubleshooting)
- [8. Inference (Testing Your Model)](#8-inference-testing-your-model)

---

## 1. Clone the Repository

> **Note:** Use your **own fork** if you’ve made code or requirements.txt changes!

```bash
git clone https://github.com/sallout/XTTSV2Finetuning.git
cd XTTSV2Finetuning
```
## 2. Table of Contents
> Make sure you’re using a Linux/Colab/RunPod environment. These are the compatible package versions:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 transformers==4.31.0
pip install tokenizers==0.13.3
pip install -r requirements.txt
```
## 3. Pretrained Model Download
> Execute the following command to download the pretrained model:
```bash
python download_checkpoint.py --output_path checkpoints/
```
> make sure you are inside XTTSV2Finetuning.

## 4. Prepare Your Dataset
```bash
For Example:
iraqi1/
  wavs/
    clip_0000.wav
    clip_0001.wav
    ...
  metadata.csv
```
> metadata.csv format:
```bash
audio_file|text|speaker
wavs/clip_0000.wav|مرحبا|iraqi_speaker
wavs/clip_0001.wav|أهلاً|iraqi_speaker
...
```

## 5. Start Training
> Replace with your actual paths as needed:

```bash
python train_gpt_xtts.py \
  --output_path ./outputs \
  --metadatas ./pretrained/iraqi1/metadata_pipe_fixed.csv,./pretrained/iraqi1/metadata_pipe_fixed.csv,ar \
  --num_epochs 40 \
  --batch_size 1
```
--output_path: Where outputs/checkpoints are saved.
--metadatas: train_csv,eval_csv, language (use the same file for both if no separate eval set).
--num_epochs: Number of epochs to train.
--batch_size: Set lower if you run out of GPU memory.

## 6. Troubleshooting
- Column errors:
Your CSV must use | as a separator and columns named exactly audio_file|text|speaker.

- Files not found:
audio_file values must be like wavs/clip_0000.wav, not with / at the start.

- CUDA out of memory:
Lower your batch size or use a larger GPU.

- Dependency errors:
Ensure your requirements match the repo and you installed the right versions.


## References
- [XTTSv2 Official Repo](https://github.com/coqui-ai/TTS)
- [XTTSv2 Hugging Face Model](https://huggingface.co/coqui/XTTS-v2)
- [Fine-tuning Template Used - XTTSv2-Finetuning-for-New-Languages ](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages)
