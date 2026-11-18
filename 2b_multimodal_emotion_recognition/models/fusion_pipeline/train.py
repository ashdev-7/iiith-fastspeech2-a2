"""Multimodal (speech + text) emotion recognition - train.

Speech branch : WavLM-Base+ -> layer-weighted proj -> 2-layer BiGRU -> attentive pooling  (temporal)
Text branch   : DistilRoBERTa -> layer-weighted proj -> BiGRU -> attentive pooling        (contextual)
Fusion        : gated multimodal unit (default) | concat | bidirectional cross-attention
Classifier    : 2-layer MLP; trained end-to-end with modality dropout

    python models/fusion_pipeline/train.py                                   # gated, GT transcripts
    python models/fusion_pipeline/train.py --fusion xattn
    python models/fusion_pipeline/train.py --transcripts asr
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.engine import base_parser, iter_configs, train_run  # noqa: E402
from common.results import summarise  # noqa: E402

ap = base_parser(__doc__)
ap.add_argument("--fusion", choices=["gated", "concat", "xattn"], default="gated")
ap.add_argument("--transcripts", choices=["gt", "asr"], default="gt")
ap.add_argument("--speech_features", choices=["wavlm", "logmel"], default="wavlm")
ap.set_defaults(modality_dropout=0.2)
args = ap.parse_args()
variant = f"{args.fusion}_{args.speech_features}_{args.transcripts}" + ("_spknorm" if args.spk_norm else "")
extra = {"fusion": args.fusion, "transcripts": args.transcripts, "speech_features": args.speech_features}
res = [train_run(cfg) for cfg in iter_configs(args, "fusion", variant, extra)]
summarise("fusion", variant, "train")
