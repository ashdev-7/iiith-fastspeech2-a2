"""Speech-only emotion recognition - train.

Preprocessing : 16 kHz, silence trim, peak norm            (common/audio.py)
Features      : frozen WavLM-Base+ hidden states, 7 of 13 layers at 40 ms (common/features.py)
                [ablation: all 13 layers at 20 ms, --features wavlm_full]
                [baseline: 80-d log-Mel, --features logmel]
Temporal      : layer-weighted projection -> 2-layer BiGRU -> attentive pooling
                [capacity study: --head attnpool = attentive statistics pooling, no recurrence]
Classifier    : 2-layer MLP

    python models/speech_pipeline/train.py                       # all protocols / seeds
    python models/speech_pipeline/train.py --protocol si --fold 0 --seed 0
    python models/speech_pipeline/train.py --features logmel
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.engine import base_parser, iter_configs, speech_variant, train_run  # noqa: E402
from common.results import summarise  # noqa: E402

ap = base_parser(__doc__)
ap.add_argument("--features", choices=["wavlm", "wavlm_full", "logmel"], default="wavlm")
args = ap.parse_args()
variant = speech_variant(args)
res = [train_run(cfg) for cfg in iter_configs(args, "speech", variant, {"speech_features": args.features})]
summarise("speech", variant, "train")
