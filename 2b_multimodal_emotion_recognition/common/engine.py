"""Training / evaluation loop shared by the speech, text and fusion pipelines."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from .config import EMOTIONS, NUM_CLASSES, PROTOCOLS, RUNS_DIR
from .data import load_index, make_split, run_name
from .features import VarLenStore, speaker_stats
from .modules import MODELS

DEFAULTS = dict(dim=256, dropout=0.2, lr=1e-3, weight_decay=1e-2, batch_size=32, epochs=40,
                patience=8, label_smoothing=0.05, modality_dropout=0.0, preload=False, skip_existing=True,
                spk_norm=False, layer_wd=-1.0, layer_lr_mult=1.0, head="bigru")
# layer_wd / layer_lr_mult: weight decay and learning-rate multiplier for the softmax layer-weight
# logits only (ablation). layer_wd < 0 means "same as weight_decay" (the setting of all main runs).
# head: speech temporal head, "bigru" (main) or "attnpool" (attentive statistics pooling, no recurrence;
# head-capacity study, analysis/PREREGISTRATION_si_capacity.md).


# ---------------------------------------------------------------- data
class FeatureDataset(Dataset):
    def __init__(self, rows: np.ndarray, labels: np.ndarray, stores: dict[str, VarLenStore], preload: bool,
                 norm: dict | None = None):
        """norm: {modality: (speaker_per_row, {speaker: (mean, std)})} -> speaker-level z-normalisation."""
        self.rows, self.labels, self.stores, self.norm = rows, labels, stores, norm or {}
        self.cache = None
        if preload:  # pull this split into RAM once (fp16 tensors) -> no disk I/O during epochs
            self.cache = {k: [torch.from_numpy(s[int(i)]) for i in rows] for k, s in stores.items()}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        item = {"idx": int(self.rows[j]), "label": int(self.labels[self.rows[j]])}
        r = int(self.rows[j])
        for k, s in self.stores.items():
            x = self.cache[k][j] if self.cache is not None else torch.from_numpy(s[r])
            if k in self.norm:
                spk, stats = self.norm[k]
                mean, std = stats[spk[r]]
                x = ((x.float() - torch.from_numpy(mean)) / torch.from_numpy(std)).half()
            item[k] = x
        return item


def collate(batch, modalities):
    """Pad variable-length (T, L, D) fp16 features; cast to fp32 happens on the device."""
    out = {"idx": torch.tensor([b["idx"] for b in batch]),
           "label": torch.tensor([b["label"] for b in batch])}
    for m in modalities:
        lens = torch.tensor([len(b[m]) for b in batch])
        out[m] = torch.nn.utils.rnn.pad_sequence([b[m] for b in batch], batch_first=True)
        out[f"{m}_len"] = lens
        out[f"{m}_mask"] = torch.arange(int(lens.max()))[None, :] < lens[:, None]
    return out


def open_stores(cfg, overrides: dict | None = None) -> dict[str, VarLenStore]:
    """overrides: {modality: store name}, e.g. noisy-audio features at test time."""
    overrides = overrides or {}
    stores = {}
    if cfg["pipeline"] in ("speech", "fusion"):
        stores["speech"] = VarLenStore(overrides.get("speech", cfg["speech_features"]))
    if cfg["pipeline"] in ("text", "fusion"):
        stores["text"] = VarLenStore(overrides.get("text", f"text_{cfg['transcripts']}"))
    return stores


def norm_spec(cfg, stores, df) -> dict:
    if not cfg.get("spk_norm") or "speech" not in stores:
        return {}
    spk = df.speaker.values
    return {"speech": (spk, speaker_stats(stores["speech"].name, spk))}


def model_meta(stores):
    meta = {}
    for k, s in stores.items():
        meta[f"{k}_layers"], meta[f"{k}_dim"] = s.num_layers, s.dim
    return meta


# ---------------------------------------------------------------- utils
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_dir(cfg) -> Path:
    return RUNS_DIR / cfg["pipeline"] / cfg["variant"] / run_name(cfg["protocol"], cfg["seed"], cfg["fold"])


def to_device(b, device):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            out[k] = v.float() if v.dtype == torch.float16 else v
        else:
            out[k] = v
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, idxs, embs, extras = [], [], [], {}, {}
    for b in loader:
        b = to_device(b, device)
        logits, emb, ext = model(b)
        probs.append(logits.softmax(-1).cpu())
        labels.append(b["label"].cpu())
        idxs.append(b["idx"].cpu())
        for k, v in emb.items():
            embs.setdefault(k, []).append(v.cpu())
        if "gate" in ext:
            extras.setdefault("gate", []).append(ext["gate"].cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    return (cat(probs), cat(labels), cat(idxs), {k: cat(v) for k, v in embs.items()},
            {k: cat(v) for k, v in extras.items()})


def metrics(y, p) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "per_class_f1": dict(zip(EMOTIONS, map(float, f1_score(y, p, average=None, labels=range(NUM_CLASSES))))),
        "confusion": confusion_matrix(y, p, labels=range(NUM_CLASSES)).tolist(),
    }


# ---------------------------------------------------------------- train / test
def train_run(cfg: dict) -> dict:
    out = run_dir(cfg)
    if cfg.get("skip_existing", True) and (out / "train_summary.json").exists() and (out / "best.pt").exists():
        print(f"[train] {cfg['pipeline']}/{cfg['variant']}/{out.name}: already trained, skipping "
              "(pass --skip_existing false to retrain)")
        return json.loads((out / "train_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_index()
    split = make_split(df, cfg["protocol"], cfg["seed"], cfg["fold"])
    stores = open_stores(cfg)
    labels = df.label.values
    mods = list(stores)
    norm = norm_spec(cfg, stores, df)
    dl = lambda rows, shuffle: DataLoader(FeatureDataset(rows, labels, stores, cfg["preload"], norm),
                                          batch_size=cfg["batch_size"], shuffle=shuffle,
                                          collate_fn=lambda b: collate(b, mods))
    tr, va = dl(split["train"], True), dl(split["val"], False)

    model = MODELS[cfg["pipeline"]](model_meta(stores), cfg).to(device)
    lw = [p for n, p in model.named_parameters() if n.endswith("layer_logits")]
    rest = [p for n, p in model.named_parameters() if not n.endswith("layer_logits")]
    layer_wd = cfg.get("layer_wd", -1.0)
    groups = [{"params": rest, "lr": cfg["lr"]}]
    if lw:
        groups.append({"params": lw, "lr": cfg["lr"] * cfg.get("layer_lr_mult", 1.0),
                       "weight_decay": cfg["weight_decay"] if layer_wd < 0 else layer_wd})
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], epochs=cfg["epochs"],
                                                steps_per_epoch=len(tr), pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    best, best_key, bad, log = None, (-1, -1e9), 0, []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for b in tr:
            b = to_device(b, device)
            logits, _, _ = model(b)
            loss = crit(logits, b["label"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b["label"])
            n += len(b["label"])
        probs, y, _, _, _ = predict(model, va, device)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(probs) + 1e-9), torch.from_numpy(y)))
        val_acc = float((probs.argmax(1) == y).mean())
        log.append(dict(epoch=ep, train_loss=tot / n, val_loss=val_loss, val_acc=val_acc))
        print(f"  ep {ep:02d} train_loss={tot / n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        # model selection: higher val accuracy, ties broken by val loss; a loss-only
        # improvement must exceed min_delta, otherwise saturated val accuracy (common on
        # TESS) would keep resetting patience on negligible loss changes.
        key = (val_acc, -val_loss)
        improved = val_acc > best_key[0] or (val_acc == best_key[0] and -val_loss > best_key[1] + 1e-3)
        if improved:
            best_key, bad = key, 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    torch.save({"state_dict": best, "cfg": cfg, "meta": model_meta(stores)}, out / "best.pt")
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    summary = dict(best_val_acc=best_key[0], best_val_loss=-best_key[1], epochs_run=len(log),
                   train_time_s=round(time.time() - t0, 1))
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] {cfg['pipeline']}/{cfg['variant']}/{out.name}: val_acc={best_key[0]:.4f} "
          f"({len(log)} ep, {summary['train_time_s']}s)")
    return summary


def load_model(cfg):
    ckpt = torch.load(run_dir(cfg) / "best.pt", map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODELS[cfg["pipeline"]](ckpt["meta"], ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, device


def evaluate(cfg: dict, store_overrides: dict | None = None, model=None):
    """Run a trained checkpoint on the test split of its protocol/seed/fold.
    `store_overrides` swaps the feature source (e.g. noisy audio) without retraining."""
    if model is None:
        model, device = load_model(cfg)
    else:
        device = next(model.parameters()).device
    df = load_index()
    split = make_split(df, cfg["protocol"], cfg["seed"], cfg["fold"])
    stores = open_stores(cfg, store_overrides)
    mods = list(stores)
    loader = DataLoader(FeatureDataset(split["test"], df.label.values, stores, False, norm_spec(cfg, stores, df)),
                        batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, mods))
    return predict(model, loader, device), model


def test_run(cfg: dict) -> dict:
    out = run_dir(cfg)
    df = load_index()
    (probs, y, idx, embs, extras), model = evaluate(cfg)
    pred = probs.argmax(1)
    m = metrics(y, pred)
    # learned layer weights (which encoder depth the model relies on)
    for name, mod in model.named_modules():
        if hasattr(mod, "layer_weights"):
            m.setdefault("layer_weights", {})[name] = mod.layer_weights().detach().cpu().tolist()
    if "gate" in extras:
        m["mean_gate_speech"] = float(extras["gate"].mean())
    (out / "test_metrics.json").write_text(json.dumps(m, indent=2))
    pred_df = df.iloc[idx][["uid", "speaker", "word", "emotion", "transcript"]].copy()
    pred_df["true"] = [EMOTIONS[i] for i in y]
    pred_df["pred"] = [EMOTIONS[i] for i in pred]
    for i, e in enumerate(EMOTIONS):
        pred_df[f"p_{e}"] = probs[:, i]
    if "gate" in extras:
        pred_df["gate_speech"] = extras["gate"].mean(1)
    pred_df.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "test_embeddings.npz", idx=idx, label=y, **embs)
    print(f"[test]  {cfg['pipeline']}/{cfg['variant']}/{out.name}: acc={m['accuracy']:.4f} "
          f"macroF1={m['macro_f1']:.4f}")
    return m


# ---------------------------------------------------------------- CLI helpers
def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--protocol", choices=list(PROTOCOLS) + ["all"], default="all",
                    help="sd = speaker-dependent word-disjoint (5 seeds); si = leave-one-speaker-out (2 folds x 3 seeds)")
    ap.add_argument("--seed", type=int, default=None, help="run a single seed (default: all seeds of the protocol)")
    ap.add_argument("--fold", type=int, default=None, help="si fold: 0 = train OAF/test YAF, 1 = train YAF/test OAF")
    ap.add_argument("--si_seeds", type=int, default=None,
                    help="number of SI seeds per fold (default: the protocol's 3; the capacity study uses 10)")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    return ap


def iter_configs(args, pipeline: str, variant: str, extra: dict):
    protocols = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    for proto in protocols:
        runs = PROTOCOLS[proto]["runs"]
        if proto == "si" and getattr(args, "si_seeds", None):
            runs = [(seed, fold) for fold in (0, 1) for seed in range(args.si_seeds)]
        for seed, fold in runs:
            if args.seed is not None and seed != args.seed:
                continue
            if args.fold is not None and fold != args.fold:
                continue
            cfg = {k: getattr(args, k) for k in DEFAULTS}
            cfg.update(extra, pipeline=pipeline, variant=variant, protocol=proto, seed=seed, fold=fold)
            yield cfg


def speech_variant(args) -> str:
    """Run-folder name of a speech configuration (shared by train.py and test.py)."""
    d = DEFAULTS
    return (args.features + ("_spknorm" if args.spk_norm else "")
            + (f"_lw{args.layer_wd:g}" if args.layer_wd >= 0 else "")
            + (f"x{args.layer_lr_mult:g}" if args.layer_lr_mult != 1 else "")
            + (f"_{args.head}" if args.head != d["head"] else "")
            + (f"_d{args.dim}" if args.dim != d["dim"] else "")
            + (f"_do{args.dropout:g}" if args.dropout != d["dropout"] else "")
            + (f"_wd{args.weight_decay:g}" if args.weight_decay != d["weight_decay"] else ""))
