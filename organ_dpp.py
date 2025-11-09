# train_organ_dpp_colab_A_C2.py
"""
ORGAN-DPP (A + C2) — Colab-ready training script
- DPP-first curriculum, validity + QED ramp (C2)
- AMP mixed precision, NaN protection, gradient clipping, checkpointing
- Expects: data/{train.txt,test.txt,vocab.txt} OR pass via CLI
"""

import os
import math
import random
import time
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# -------------------------
# CUDA optimization settings
# -------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# -------------------------
# Config / hyperparameters
# -------------------------
def get_default_config():
    return {
        "data_dir": ".",
        "train_file": "train.txt",
        "test_file": "test.txt",
        "vocab_file": "vocab.txt",
        "save_dir": "checkpoints",
        "device": None,
        "batch_size": 512,
        "seq_max_len": 80,
        "embed_size": 96,
        "hidden_size": 384,
        "num_layers": 1,
        "dropout": 0.1,
        "lr": 2e-4,
        "epochs": 20,
        "dpp_k": 32,
        "dpp_feature_dim": 512,
        "save_every": 5,
        "grad_clip": 1.0,
        "seed": 42,
        "invalid_penalty": 0.3,
        "num_workers": 2,  # Reduced for stability
        "pin_memory": True,
        "amp": True,
        "accumulate_grad_batches": 1,
        "fast_eval_batches": 2,
        "print_every_batches": 50,
        "reward_normalize": True  # NEW: normalize rewards
    }

config = get_default_config()

# reproducibility
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])

# -------------------------
# RDKit optional utils + logging suppression
# -------------------------
USE_RDKIT = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, QED
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from rdkit import DataStructs
except Exception:
    USE_RDKIT = False
    print("[WARN] RDKit not available; using fallback fingerprint and no validity checks.")

def mol_validity(smiles):
    if not USE_RDKIT:
        return True
    try:
        m = Chem.MolFromSmiles(smiles)
        return m is not None
    except Exception:
        return False

def morgan_fingerprint_array(smiles, n_bits=1024, radius=2):
    """Generate Morgan fingerprint with proper error handling"""
    if USE_RDKIT:
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return np.zeros(n_bits, dtype=np.float32)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return np.zeros(n_bits, dtype=np.float32)
    else:
        # Fallback hash-based fingerprint
        vec = np.zeros(n_bits, dtype=np.float32)
        tokens = list(smiles)
        for i in range(len(tokens)):
            ng = tokens[i:i+3]
            key = "".join(ng)
            h = abs(hash(key)) % n_bits
            vec[h] += 1.0
        l2 = np.linalg.norm(vec)
        if l2 > 0: 
            vec /= l2
        return vec

# -------------------------
# Vocab helpers
# -------------------------
def load_vocab(path):
    tokens = [t.strip() for t in open(path).read().splitlines() if t.strip()]
    stoi = {t:i for i,t in enumerate(tokens)}
    itos = {i:t for i,t in enumerate(tokens)}
    return tokens, stoi, itos

# -------------------------
# Dataset + Tokenizer
# -------------------------
class SMILESDataset(Dataset):
    def __init__(self, filepath, stoi, seq_max_len=120):
        self.lines = [l.strip() for l in open(filepath).read().splitlines() if l.strip()]
        self.stoi = stoi
        self.maxlen = seq_max_len

    def __len__(self):
        return len(self.lines)

    def encode_tokens(self, token_list):
        ids = [self.stoi.get(t, self.stoi.get("<unk>", 3)) for t in token_list]
        if len(ids) > self.maxlen:
            ids = ids[:self.maxlen]
            ids[-1] = self.stoi.get("<eos>", 2)
        ids = ids + [self.stoi.get("<pad>", 0)] * (self.maxlen - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        line = self.lines[idx]
        toks = line.split()
        ids = self.encode_tokens(toks)
        raw = "".join([t for t in toks if t not in ("<bos>","<eos>","<pad>")])
        return ids, raw

# -------------------------
# Models
# -------------------------
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout if num_layers > 1 else 0, 
                           batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hx=None):
        emb = self.embed(x)
        out, hx = self.lstm(emb, hx)
        logits = self.output(out)
        return logits, hx

    @torch.no_grad()
    def sample(self, batch_size, seq_len, temperature=1.0, device=torch.device("cpu")):
        self.eval()
        inputs = torch.full((batch_size,1), BOS, dtype=torch.long, device=device)
        hx = None
        samples = []
        
        for t in range(seq_len):
            logits, hx = self.forward(inputs, hx)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            samples.append(next_token)
            inputs = torch.cat([inputs, next_token], dim=1)
            
        sampled = torch.cat(samples, dim=1)
        self.train()
        return sampled

class CNNDiscriminator(nn.Module):
    def __init__(self, vocab_size, embed_size=64, conv_channels=(64,128,128)):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        layers = []
        in_ch = embed_size
        ks = (3,5,7)
        for out_ch, k in zip(conv_channels, ks):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        emb = self.embed(x).permute(0,2,1)  # (B, embed, seq)
        out = self.conv(emb)
        out = self.pool(out).squeeze(-1)  # (B, conv_channels)
        logits = self.fc(out).squeeze(-1)  # (B,)
        return torch.sigmoid(logits)

# -------------------------
# DPP sampler (small-d Gram trick) - IMPROVED
# -------------------------
def feature_dpp_sample(feature_matrix, k):
    """
    Improved DPP sampling with better numerical stability
    """
    n, d = feature_matrix.shape
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    
    X = feature_matrix.copy().astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    
    # Small Gram matrix trick
    G = X.T.dot(X) + np.eye(d) * 1e-6  # Added regularization
    
    try:
        w, V = np.linalg.eigh(G)
    except np.linalg.LinAlgError:
        # Fallback to random sampling if eigendecomposition fails
        return list(np.random.choice(n, size=min(k, n), replace=False))
    
    U = X.dot(V)
    
    # Select eigenvalues
    selected = []
    for i, wi in enumerate(w):
        prob = float(wi / (1.0 + wi))
        if np.random.rand() < prob:
            selected.append(i)
    
    if len(selected) == 0:
        selected = [int(np.argmax(w))]
    
    Vsel = U[:, selected]
    chosen = []
    
    # Gram-Schmidt sampling
    for _ in range(min(k, n)):
        if Vsel.shape[1] == 0:
            break
            
        probs = np.sum(Vsel**2, axis=1)
        probs = np.maximum(probs, 0.0)
        
        if probs.sum() <= 1e-10:
            break
            
        probs = probs / probs.sum()
        
        try:
            i = np.random.choice(n, p=probs)
        except ValueError:
            break
            
        if i in chosen:
            break
            
        chosen.append(i)
        
        vi = Vsel[i:i+1, :].copy()
        vi_norm = np.linalg.norm(vi)
        
        if vi_norm < 1e-10:
            break
            
        vi = vi / vi_norm
        Vsel = Vsel - (Vsel.dot(vi.T) * vi)
    
    return chosen if len(chosen) > 0 else list(range(min(k, n)))

# -------------------------
# Utilities
# -------------------------
def seq_ids_to_smiles(ids, itos):
    toks = [itos.get(int(i), "<unk>") for i in ids if int(i) not in [PAD, BOS, EOS]]
    return "".join(toks)

def normalize_rewards(rewards):
    """Normalize rewards to have mean 0 and std 1"""
    mean = np.mean(rewards)
    std = np.std(rewards)
    if std < 1e-8:
        return rewards - mean
    return (rewards - mean) / (std + 1e-8)

# -------------------------
# CLI args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=config["data_dir"])
parser.add_argument("--epochs", type=int, default=config["epochs"])
parser.add_argument("--batch_size", type=int, default=config["batch_size"])
parser.add_argument("--seq_max_len", type=int, default=config["seq_max_len"])
parser.add_argument("--amp", type=int, choices=[0,1], default=1)
parser.add_argument("--resume", type=str, default=None)
args, unknown = parser.parse_known_args()

config["data_dir"] = args.data_dir
config["epochs"] = args.epochs
config["batch_size"] = args.batch_size
config["seq_max_len"] = args.seq_max_len
config["amp"] = bool(args.amp)

# -------------------------
# Load vocab
# -------------------------
vocab_path = os.path.join(config["data_dir"], config["vocab_file"])
if not os.path.exists(vocab_path):
    raise FileNotFoundError(f"vocab file not found: {vocab_path}")

tokens, stoi, itos = load_vocab(vocab_path)
vocab_size = len(tokens)
PAD = stoi.get("<pad>", 0)
BOS = stoi.get("<bos>", 1)
EOS = stoi.get("<eos>", 2)
UNK = stoi.get("<unk>", 3)

print(f"Vocabulary size: {vocab_size}")
print(f"Special tokens: PAD={PAD}, BOS={BOS}, EOS={EOS}, UNK={UNK}")

# -------------------------
# Setup datasets + loaders
# -------------------------
train_path = os.path.join(config["data_dir"], config["train_file"])
test_path  = os.path.join(config["data_dir"], config["test_file"])

if not os.path.exists(train_path):
    raise FileNotFoundError(f"train.txt not found: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"test.txt not found: {test_path}")

train_ds = SMILESDataset(train_path, stoi, seq_max_len=config["seq_max_len"])
test_ds  = SMILESDataset(test_path, stoi, seq_max_len=config["seq_max_len"])

print(f"Training samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")

train_loader = DataLoader(
    train_ds, 
    batch_size=config["batch_size"], 
    shuffle=True, 
    drop_last=True,
    num_workers=config["num_workers"], 
    pin_memory=config["pin_memory"],
    persistent_workers=True if config["num_workers"] > 0 else False
)

test_loader = DataLoader(
    test_ds, 
    batch_size=config["batch_size"], 
    shuffle=False,
    num_workers=max(0, config["num_workers"]//2), 
    pin_memory=config["pin_memory"]
)

# -------------------------
# Instantiate models / optimizers / losses
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

G = LSTMGenerator(
    vocab_size, 
    config["embed_size"], 
    config["hidden_size"], 
    config["num_layers"], 
    config["dropout"]
).to(device)

D = CNNDiscriminator(vocab_size).to(device)

print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")

opt_G = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(0.9, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(0.9, 0.999))

# Loss functions
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss(ignore_index=PAD)

# AMP scaler
scaler = torch.cuda.amp.GradScaler(enabled=(config["amp"] and device.type=="cuda"))

# -------------------------
# Curriculum helpers (A + C2)
# -------------------------
def get_stage(epoch, N):
    """
    A + C2 schedule:
    Stage1 - DPP-heavy exploration (validity penalty small)
    Stage2 - validity + QED ramp
    Stage3 - balanced refinement
    """
    s_len = max(1, N // 3)
    if epoch <= s_len:
        return {
            "stage": 1, 
            "reward_weights": (0.9, 0.1, 0.0), 
            "temp_range": (1.6, 1.2),
            "name": "DPP Exploration"
        }
    elif epoch <= 2*s_len:
        return {
            "stage": 2, 
            "reward_weights": (0.7, 0.3, 0.0), 
            "temp_range": (1.2, 0.9),
            "name": "Validity+QED Ramp"
        }
    else:
        return {
            "stage": 3, 
            "reward_weights": (0.5, 0.35, 0.15), 
            "temp_range": (0.9, 0.7),
            "name": "Balanced Refinement"
        }

def temperature_for_epoch(epoch, epoch_start, epoch_end, temp_range):
    """Cosine annealing temperature schedule"""
    t = (epoch - epoch_start) / max(1, (epoch_end - epoch_start))
    t = min(max(t, 0.0), 1.0)
    tmin, tmax = temp_range[1], temp_range[0]
    return tmin + 0.5*(tmax - tmin)*(1 + math.cos(math.pi * t))

# -------------------------
# Reward functions
# -------------------------
def reward_validity(smiles_list):
    """Binary validity reward"""
    return np.array([1.0 if mol_validity(s) else 0.0 for s in smiles_list], dtype=np.float32)

def reward_qed_approx(smiles_list):
    """QED-based reward"""
    if USE_RDKIT:
        try:
            out = []
            for s in smiles_list:
                try:
                    m = Chem.MolFromSmiles(s)
                    out.append(QED.qed(m) if m is not None else 0.0)
                except Exception:
                    out.append(0.0)
            return np.array(out, dtype=np.float32)
        except Exception:
            pass
    
    # Fallback: length-based heuristic
    out = []
    for s in smiles_list:
        L = len(s)
        score = 1.0 - abs(L - 30)/30.0
        out.append(max(0.0, min(1.0, score)))
    return np.array(out, dtype=np.float32)

def reward_sa_approx(smiles_list):
    """Synthetic accessibility approximation"""
    out = []
    for s in smiles_list:
        L = len(s)
        score = 1.0 - (L / 200.0)
        out.append(max(0.0, min(1.0, score)))
    return np.array(out, dtype=np.float32)

# -------------------------
# Training loop
# -------------------------
os.makedirs(config["save_dir"], exist_ok=True)
global_step = 0
n_epochs = config["epochs"]

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

for epoch in range(1, n_epochs+1):
    G.train()
    D.train()
    
    # Get curriculum configuration
    stage_cfg = get_stage(epoch, n_epochs)
    stage_idx = stage_cfg["stage"]
    total = n_epochs
    st_len = max(1, total // 3)
    stage_start = (stage_idx-1)*st_len + 1
    stage_end = min(total, stage_idx*st_len)
    temp = temperature_for_epoch(epoch, stage_start, stage_end, stage_cfg["temp_range"])

    print(f"\nEpoch {epoch}/{n_epochs} - Stage {stage_idx}: {stage_cfg['name']}")
    print(f"Temperature: {temp:.3f} | Reward weights: {stage_cfg['reward_weights']}")

    running_G_loss = 0.0
    running_D_loss = 0.0
    running_validity = 0.0
    running_qed = 0.0
    batches = 0
    start_time = time.time()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training")
    
    for batch_idx, (ids_batch, raw_smiles_batch) in loop:
        ids_batch = ids_batch.to(device)
        B = ids_batch.size(0)
        batches += 1

        # ===== GENERATOR PHASE =====
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            # Sample molecules
            sampled_ids = G.sample(B, config["seq_max_len"], temperature=temp, device=device)
            
            # Convert to SMILES
            sampled_smiles = []
            for i in range(B):
                s = seq_ids_to_smiles(sampled_ids[i].cpu().numpy(), itos)
                sampled_smiles.append(s if len(s) > 0 else "C")  # Fallback to carbon

            # Compute fingerprints (batched for efficiency)
            try:
                feats = np.stack([
                    morgan_fingerprint_array(s, n_bits=config["dpp_feature_dim"]) 
                    for s in sampled_smiles
                ])
            except Exception as e:
                print(f"[WARN] Fingerprint computation failed: {e}")
                feats = np.random.randn(B, config["dpp_feature_dim"]).astype(np.float32)

            # DPP sampling
            try:
                chosen_idx = feature_dpp_sample(feats, min(config["dpp_k"], B))
            except Exception as e:
                print(f"[WARN] DPP sampling failed: {e}, using random sampling")
                chosen_idx = list(np.random.choice(B, size=min(config["dpp_k"], B), replace=False))

            if len(chosen_idx) == 0:
                chosen_idx = list(range(min(B, config["dpp_k"])))

            # Select diverse subset
            selected_ids = sampled_ids[chosen_idx]
            selected_smiles = [sampled_smiles[i] for i in chosen_idx]

            # Compute rewards
            r_valid = reward_validity(selected_smiles)
            r_qed = reward_qed_approx(selected_smiles)
            r_sa = reward_sa_approx(selected_smiles)

            w_valid, w_qed, w_sa = stage_cfg["reward_weights"]
            r_total = w_valid * r_valid + w_qed * r_qed + w_sa * r_sa

            # Invalid penalty (C2 mechanism)
            invalid_mask = (r_valid == 0.0)
            if invalid_mask.any():
                r_total[invalid_mask] -= config["invalid_penalty"]

            # Normalize rewards if enabled
            if config["reward_normalize"]:
                r_total = normalize_rewards(r_total)
            
            r_total = np.clip(r_total, -2.0, 2.0)

            # Prepare for loss computation
            inputs = selected_ids[:, :-1].to(device)
            targets = selected_ids[:, 1:].to(device)
            
            logits, _ = G(inputs)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)

            # Cross-entropy loss weighted by rewards
            ce = ce_loss(logits_flat, targets_flat)
            
            # Reward scaling (inverse of mean reward)
            mean_reward = float(np.mean(r_total))
            reward_scale = max(0.1, 1.0 - mean_reward)
            
            G_loss = ce * reward_scale

            # Adversarial component
            D_fake_scores = D(selected_ids.to(device))
            adv_target = torch.ones_like(D_fake_scores)
            adv_loss = bce_loss(D_fake_scores, adv_target)
            
            G_loss = G_loss + 0.1 * adv_loss

        # Backward pass for Generator
        opt_G.zero_grad()
        
        try:
            scaler.scale(G_loss).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), config["grad_clip"])
            
            # NaN check
            if torch.isnan(G_loss).any():
                print(f"[WARN] NaN in G_loss at batch {batch_idx}")
                opt_G.zero_grad()
            else:
                scaler.step(opt_G)
            
            scaler.update()
            
        except Exception as e:
            print(f"[ERROR] G backward failed: {e}")
            opt_G.zero_grad()
            scaler.update()

        # ===== DISCRIMINATOR PHASE =====
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            # Real and fake data
            real_n = min(len(chosen_idx), ids_batch.size(0))
            real_ids = ids_batch[:real_n]
            fake_ids = selected_ids.detach()

            # Discriminator predictions
            D_real = D(real_ids)
            D_fake = D(fake_ids)

            # BCE loss
            real_labels = torch.ones_like(D_real)
            fake_labels = torch.zeros_like(D_fake)
            
            loss_D = bce_loss(D_real, real_labels) + bce_loss(D_fake, fake_labels)

        # Backward pass for Discriminator
        opt_D.zero_grad()
        
        try:
            scaler.scale(loss_D).backward()
            scaler.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(D.parameters(), config["grad_clip"])
            
            # NaN check
            if torch.isnan(loss_D).any():
                print(f"[WARN] NaN in D_loss at batch {batch_idx}")
                opt_D.zero_grad()
            else:
                scaler.step(opt_D)
            
            scaler.update()
            
        except Exception as e:
            print(f"[ERROR] D backward failed: {e}")
            opt_D.zero_grad()
            scaler.update()

        # Track metrics
        running_G_loss += float(G_loss.detach().cpu().item())
        running_D_loss += float(loss_D.detach().cpu().item())
        running_validity += float(r_valid.mean())
        running_qed += float(r_qed.mean())
        
        global_step += 1

        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            loop.set_postfix({
                "G": f"{running_G_loss/(batch_idx+1):.3f}",
                "D": f"{running_D_loss/(batch_idx+1):.3f}",
                "Valid": f"{running_validity/(batch_idx+1):.2f}",
                "QED": f"{running_qed/(batch_idx+1):.3f}"
            })

    # Epoch summary
    epoch_time = time.time() - start_time
    avg_G = running_G_loss / max(1, batches)
    avg_D = running_D_loss / max(1, batches)
    avg_valid = running_validity / max(1, batches)
    avg_qed = running_qed / max(1, batches)
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Time: {epoch_time:.1f}s")
    print(f"  G Loss: {avg_G:.4f} | D Loss: {avg_D:.4f}")
    print(f"  Validity: {avg_valid:.4f} | QED: {avg_qed:.4f}")

    # Save checkpoint
    if epoch % config["save_every"] == 0 or epoch == n_epochs:
        ckpt = {
            "epoch": epoch,
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "config": config,
            "itos": itos,
            "stoi": stoi,
            "metrics": {
                "G_loss": avg_G,
                "D_loss": avg_D,
                "validity": avg_valid,
                "qed": avg_qed
            }
        }
        fname = os.path.join(config["save_dir"], f"organ_dpp_epoch{epoch}.pt")
        torch.save(ckpt, fname)
        print(f"✓ Saved checkpoint: {fname}")

    # Evaluation
    if epoch % config["save_every"] == 0 or epoch == n_epochs:
        print("\nRunning evaluation...")
        G.eval()
        all_sampled = []
        
        with torch.no_grad():
            for _ in range(config["fast_eval_batches"]):
                sampled = G.sample(config["batch_size"], config["seq_max_len"], temperature=temp, device=device)
                for i in range(sampled.size(0)):
                    s = seq_ids_to_smiles(sampled[i].cpu().numpy(), itos)
                    if len(s) > 0:
                        all_sampled.append(s)
        
        if len(all_sampled) > 0:
            # Compute validity
            if USE_RDKIT:
                valids = [mol_validity(s) for s in all_sampled]
                valid_pct = 100.0 * sum(valids) / len(valids)
                valid_smiles = [s for s, v in zip(all_sampled, valids) if v]
            else:
                valid_pct = 100.0
                valid_smiles = all_sampled
            
            # Compute QED for valid molecules
            if len(valid_smiles) > 0:
                qed_vals = reward_qed_approx(valid_smiles)
                mean_qed = float(np.mean(qed_vals))
                
                # Compute uniqueness
                unique_smiles = set(valid_smiles)
                uniqueness_pct = 100.0 * len(unique_smiles) / len(valid_smiles)
                
                # Compute diversity (simple Tanimoto-based)
                if USE_RDKIT and len(unique_smiles) > 1:
                    try:
                        fps = []
                        for s in list(unique_smiles)[:100]:  # Sample 100 for speed
                            m = Chem.MolFromSmiles(s)
                            if m:
                                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                                fps.append(fp)
                        
                        if len(fps) > 1:
                            sims = []
                            for i in range(len(fps)):
                                for j in range(i+1, len(fps)):
                                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                                    sims.append(sim)
                            diversity = 1.0 - np.mean(sims)
                        else:
                            diversity = 0.0
                    except Exception:
                        diversity = 0.0
                else:
                    diversity = 0.0
            else:
                mean_qed = 0.0
                uniqueness_pct = 0.0
                diversity = 0.0
            
            print(f"\nValidation Results ({len(all_sampled)} samples):")
            print(f"  Validity:    {valid_pct:.2f}%")
            print(f"  Uniqueness:  {uniqueness_pct:.2f}%")
            print(f"  Mean QED:    {mean_qed:.4f}")
            print(f"  Diversity:   {diversity:.4f}")
            
            # Print sample molecules
            print(f"\nSample Generated SMILES:")
            for i, s in enumerate(all_sampled[:5]):
                valid_mark = "✓" if mol_validity(s) else "✗"
                print(f"  {i+1}. [{valid_mark}] {s}")
        else:
            print("[WARN] No molecules generated in evaluation")
        
        G.train()

print("\n" + "="*80)
print("TRAINING COMPLETED")
print("="*80)
print(f"Final model saved to: {config['save_dir']}")