"""Text-only emotion recognition - test.

Preprocessing : lower-case, symbol clean-up, RoBERTa BPE tokenisation   (common/data.py)
Features      : frozen emotion-tuned DistilRoBERTa hidden states, 7 layers (common/features.py)
Contextual    : layer-weighted projection -> BiGRU -> attentive pooling
Classifier    : 2-layer MLP

    python models/text_pipeline/test.py                         # ground-truth transcripts
    python models/text_pipeline/test.py --transcripts asr       # Whisper transcripts (extension)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.engine import base_parser, iter_configs, test_run  # noqa: E402
from common.results import summarise  # noqa: E402

ap = base_parser(__doc__)
ap.add_argument("--transcripts", choices=["gt", "asr"], default="gt")
args = ap.parse_args()
res = [test_run(cfg) for cfg in iter_configs(args, "text", args.transcripts, {"transcripts": args.transcripts})]
summarise("text", args.transcripts, "test")
