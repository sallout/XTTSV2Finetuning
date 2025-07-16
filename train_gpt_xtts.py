import os
import gc
import shutil 
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

import argparse
import torch


def create_xtts_trainer_parser():
    parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to pretrained + checkpoint model")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv_path,eval_csv_path,language")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Grad accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Max audio length")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Max text length")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Save step")

    return parser

def save_best_model(ckpt_dir):
    import torch, shutil, os
    best_loss = float("inf")
    best_ckpt = None
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            try:
                ckpt = torch.load(fpath, map_location="cpu")
                loss = ckpt.get("model_loss", None)
                loss_val = None
                if isinstance(loss, dict):
                    # Prefer eval_loss if it's a number
                    if "eval_loss" in loss and isinstance(loss["eval_loss"], (float, int)):
                        loss_val = loss["eval_loss"]
                    elif "train_loss" in loss and isinstance(loss["train_loss"], (float, int)):
                        loss_val = loss["train_loss"]
                    elif "loss" in loss and isinstance(loss["loss"], (float, int)):
                        loss_val = loss["loss"]
                    else:
                        # Try first valid numeric value
                        for v in loss.values():
                            if isinstance(v, (float, int)):
                                loss_val = v
                                break
                elif isinstance(loss, (float, int)):
                    loss_val = loss
                if loss_val is not None and isinstance(loss_val, (float, int)):
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_ckpt = fpath
                else:
                    print(f"⚠️ No valid loss found in {fname} (loss={loss}, extracted={loss_val})")
            except Exception as e:
                print(f"⚠️ Failed to process {fname}: {e}")

    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No valid checkpoint found to select as best!")

    """
    Scan all checkpoints in ckpt_dir, find the one with the lowest eval_loss (or train_loss/loss as fallback),
    and copy it to 'best_model.pth'.
    """
    best_loss = float("inf")
    best_ckpt = None
    found_any = False

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            found_any = True
            fpath = os.path.join(ckpt_dir, fname)
            try:
                ckpt = torch.load(fpath, map_location="cpu")
                loss = ckpt.get("model_loss", None)
                loss_val = None

                if isinstance(loss, dict):
                    # Pick the first available valid float/int: eval_loss, train_loss, loss, then any non-None value
                    for key in ["eval_loss", "train_loss", "loss"]:
                        v = loss.get(key)
                        if isinstance(v, (float, int)) and v is not None:
                            loss_val = v
                            break
                    # fallback: pick any first non-None value
                    if loss_val is None and len(loss) > 0:
                        for v in loss.values():
                            if isinstance(v, (float, int)) and v is not None:
                                loss_val = v
                                break
                elif isinstance(loss, (float, int)):
                    loss_val = loss
                else:
                    print(f"⚠️ model_loss in {fname} is not usable: {loss}")

                if loss_val is not None and isinstance(loss_val, (float, int)):
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_ckpt = fpath
                else:
                    print(f"⚠️ No valid loss found in {fname} (loss={loss}, extracted={loss_val})")
            except Exception as e:
                print(f"⚠️ Failed to process {fname}: {e}")

    if not found_any:
        print("\n❌ No checkpoint files found in directory:", ckpt_dir)
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No valid checkpoint found to select as best!")

    """
    Scan all checkpoints in ckpt_dir, find the one with the lowest eval_loss (or train_loss/loss as fallback),
    and copy it to 'best_model.pth'.
    """
    best_loss = float("inf")
    best_ckpt = None
    found_any = False

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            found_any = True
            fpath = os.path.join(ckpt_dir, fname)
            try:
                ckpt = torch.load(fpath, map_location="cpu")
                loss = ckpt.get("model_loss", None)
                loss_val = None

                if isinstance(loss, dict):
                    # Prefer eval_loss > train_loss > loss > any
                    if "eval_loss" in loss:
                        loss_val = loss["eval_loss"]
                    elif "train_loss" in loss:
                        loss_val = loss["train_loss"]
                    elif "loss" in loss:
                        loss_val = loss["loss"]
                    elif len(loss) > 0:
                        loss_val = list(loss.values())[0]
                elif isinstance(loss, (float, int)):
                    loss_val = loss
                else:
                    print(f"⚠️ model_loss in {fname} is not usable: {loss}")

                if loss_val is not None and isinstance(loss_val, (float, int)):
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_ckpt = fpath
                else:
                    print(f"⚠️ No valid loss found in {fname} (loss={loss}, extracted={loss_val})")
            except Exception as e:
                print(f"⚠️ Failed to process {fname}: {e}")

    if not found_any:
        print("\n❌ No checkpoint files found in directory:", ckpt_dir)
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No valid checkpoint found to select as best!")

    """
    Scan all checkpoints in ckpt_dir, find the one with the lowest eval_loss (or train_loss/loss as fallback),
    and copy it to 'best_model.pth'.
    """
    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            try:
                ckpt = torch.load(fpath, map_location="cpu")
                loss = ckpt.get("model_loss", None)
                loss_val = None

                if isinstance(loss, dict):
                    # Prefer eval_loss > train_loss > loss > any
                    if "eval_loss" in loss:
                        loss_val = loss["eval_loss"]
                    elif "train_loss" in loss:
                        loss_val = loss["train_loss"]
                    elif "loss" in loss:
                        loss_val = loss["loss"]
                    elif len(loss) > 0:
                        loss_val = list(loss.values())[0]
                elif isinstance(loss, (float, int)):
                    loss_val = loss
                else:
                    continue  # unknown format

                if loss_val is not None and isinstance(loss_val, (float, int)):
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_ckpt = fpath
            except Exception as e:
                print(f"⚠️ Failed to process {fname}: {e}")

    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

    """
    Scan all checkpoints in ckpt_dir, find the one with the lowest eval_loss (or train_loss/loss as fallback),
    and copy it to 'best_model.pth'.
    """
   
    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            try:
                ckpt = torch.load(fpath, map_location="cpu")
                loss = ckpt.get("model_loss", None)
                loss_val = None

                if isinstance(loss, dict):
                    # Prefer eval_loss > train_loss > loss > any
                    if "eval_loss" in loss:
                        loss_val = loss["eval_loss"]
                    elif "train_loss" in loss:
                        loss_val = loss["train_loss"]
                    elif "loss" in loss:
                        loss_val = loss["loss"]
                    elif len(loss) > 0:
                        loss_val = list(loss.values())[0]
                elif isinstance(loss, (float, int)):
                    loss_val = loss
                else:
                    continue  # unknown format

                if loss_val is not None and isinstance(loss_val, (float, int)):
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_ckpt = fpath
            except Exception as e:
                print(f"⚠️ Failed to process {fname}: {e}")

    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")


    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(fpath, map_location="cpu")
            loss = ckpt.get("model_loss")
            # Safely extract the float loss value
            if isinstance(loss, dict):
                if "eval_loss" in loss:
                    loss_val = loss["eval_loss"]
                elif "train_loss" in loss:
                    loss_val = loss["train_loss"]
                elif "loss" in loss:
                    loss_val = loss["loss"]
                else:
                    loss_val = list(loss.values())[0]
            elif isinstance(loss, (float, int)):
                loss_val = loss
            else:
                print(f"Skipping {fname} (unknown model_loss type: {type(loss)})")
                continue
            if isinstance(loss_val, (float, int)) and loss_val < best_loss:
                best_loss = loss_val
                best_ckpt = fpath
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

    """Scan all checkpoints, find lowest eval_loss (or loss), save as best_model.pth"""
    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(fpath, map_location="cpu")
            loss = ckpt.get("model_loss")
            loss_val = None
            if isinstance(loss, dict):
                # Try to pick best loss metric
                if "eval_loss" in loss:
                    loss_val = loss["eval_loss"]
                elif "loss" in loss:
                    loss_val = loss["loss"]
                elif "train_loss" in loss:
                    loss_val = loss["train_loss"]
                else:
                    # fallback: pick first value
                    loss_val = list(loss.values())[0]
            elif isinstance(loss, (float, int)):
                loss_val = loss
            if isinstance(loss_val, (float, int)) and loss_val < best_loss:
                best_loss = loss_val
                best_ckpt = fpath
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

    """Scan all checkpoints, find lowest eval_loss, save as best_model.pth"""
    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(fpath, map_location="cpu")
            loss = ckpt.get("model_loss")
            # Safely extract the float loss value
            if isinstance(loss, dict):
                # Prefer eval_loss, else train_loss, else pick first value
                if "eval_loss" in loss:
                    loss_val = loss["eval_loss"]
                elif "train_loss" in loss:
                    loss_val = loss["train_loss"]
                elif "loss" in loss:
                    loss_val = loss["loss"]
                else:
                    loss_val = list(loss.values())[0]
            elif isinstance(loss, (float, int)):
                loss_val = loss
            else:
                print(f"Skipping {fname} (unknown model_loss type: {type(loss)})")
                continue
            if isinstance(loss_val, (float, int)) and loss_val < best_loss:
                best_loss = loss_val
                best_ckpt = fpath
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

    """Scan all checkpoints, find lowest eval_loss, save as best_model.pth"""
    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(fpath, map_location="cpu")
            loss = ckpt.get("model_loss")
            # Safely extract the float loss value
            if isinstance(loss, dict):
                # Prefer eval_loss, else train_loss, else skip
                if "eval_loss" in loss:
                    loss_val = loss["eval_loss"]
                elif "train_loss" in loss:
                    loss_val = loss["train_loss"]
                else:
                    print(f"Skipping {fname} (no eval_loss or train_loss)")
                    continue
            elif isinstance(loss, (float, int)):
                loss_val = loss
            else:
                print(f"Skipping {fname} (unknown model_loss type: {type(loss)})")
                continue
            # Only now compare!
            if loss_val < best_loss:
                best_loss = loss_val
                best_ckpt = fpath
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(fpath, map_location="cpu")
            if "model_loss" in ckpt:
                loss_val = ckpt["model_loss"]
                if isinstance(loss_val, dict):
                    if "eval_loss" in loss_val:
                        loss_val = loss_val["eval_loss"]
                    elif "loss" in loss_val:
                        loss_val = loss_val["loss"]
                    else:
                        loss_val = list(loss_val.values())[0]
                if isinstance(loss_val, (float, int)):
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_ckpt = fpath
                else:
                    print(f"[Warning] Could not interpret model_loss in {fname}: {repr(loss_val)}")
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (eval_loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(fpath, map_location="cpu")
            if "model_loss" in ckpt:
                loss_val = ckpt["model_loss"]
                if isinstance(loss_val, dict):
                    # Debug: print keys found
                    print(f"[Debug] model_loss dict keys: {list(loss_val.keys())} in {fname}")
                    # Try to get the main loss value
                    if "loss" in loss_val:
                        loss_val = loss_val["loss"]
                    else:
                        # Pick the first value as fallback
                        loss_val = list(loss_val.values())[0]
                if isinstance(loss_val, (float, int)):
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_ckpt = fpath
                else:
                    print(f"[Warning] Could not interpret model_loss in {fname}: {repr(loss_val)}")
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

    """Scan all checkpoints, find lowest model_loss, save as best_model.pth"""
    best_loss = float("inf")
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pth"):
            fpath = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(fpath, map_location="cpu")
            if "model_loss" in ckpt:
                loss = ckpt["model_loss"]
                if loss < best_loss:
                    best_loss = loss
                    best_ckpt = fpath
    if best_ckpt:
        shutil.copy2(best_ckpt, os.path.join(ckpt_dir, "best_model.pth"))
        print(f"\n✅ Best checkpoint is {os.path.basename(best_ckpt)} (loss {best_loss:.4f})\nSaved as best_model.pth")
    else:
        print("\n❌ No checkpoint found to select as best!")

def train_gpt(metadatas, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, max_text_length, lr, weight_decay, save_step):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    OUT_PATH = output_path

    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
    START_WITH_EVAL = False
    BATCH_SIZE = batch_size
    GRAD_ACUMM_STEPS = grad_acumm

    DATASETS_CONFIG_LIST = []
    for metadata in metadatas:
        train_csv, eval_csv, language = metadata.split(",")
        print(train_csv, eval_csv, language)

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )
        DATASETS_CONFIG_LIST.append(config_dataset)

    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))

    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files(
            [XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files(
            [XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=11025,
        debug_loading_failures=False,
        max_wav_length=max_audio_length,
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    config = GPTTrainerConfig()
    config.load_json(XTTS_CONFIG_FILE)
    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = """
        GPT XTTS training
        """,
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # === [NEW] Find and save best model after training
    trainer_out_path = trainer.output_path
    save_best_model(trainer_out_path)
    # ===

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]
    
    trainer_out_path = trainer.output_path

    del model, trainer, train_samples, eval_samples
    gc.collect()

    return trainer_out_path

if __name__ == "__main__":
    parser = create_xtts_trainer_parser()
    args = parser.parse_args()

    trainer_out_path = train_gpt(
        metadatas=args.metadatas,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_acumm,
        weight_decay=args.weight_decay,
        lr=args.lr,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        save_step=args.save_step
    )

    print(f"Checkpoint saved in dir: {trainer_out_path}")
