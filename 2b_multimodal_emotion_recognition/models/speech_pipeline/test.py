"""Speech-only emotion recognition - test.

Preprocessing : 16 kHz, silence trim, peak norm            (common/audio.py)
Features      : frozen WavLM-Base+ hidden states, 7 of 13 layers at 40 ms (common/features.py)
                [ablation: all 13 layers at 20 ms, --features wavlm_full]
                [baseline: 80-d log-Mel, --features logmel]
Temporal      : layer-weighted projection -> 2-layer BiGRU -> attentive pooling
Classifier    : 2-layer MLP

    python models/speech_pipeline/test.py                       # all protocols / seeds
    python models/speech_pipeline/test.py --protocol si --fold 0 --seed 0
    python models/speech_pipeline/test.py --features logmel
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.engine import base_parser, iter_configs, test_run  # noqa: E402
from common.results import summarise  # noqa: E402

ap = base_parser(__doc__)
ap.add_argument("--features", choices=["wavlm", "wavlm_full", "logmel"], default="wavlm")
args = ap.parse_args()
variant = (args.features + ("_spknorm" if args.spk_norm else "")
           + (f"_lw{args.layer_wd:g}" if args.layer_wd >= 0 else "")
           + (f"x{args.layer_lr_mult:g}" if args.layer_lr_mult != 1 else ""))
res = [test_run(cfg) for cfg in iter_configs(args, "speech", variant, {"speech_features": args.features})]
summarise("speech", variant, "test")
