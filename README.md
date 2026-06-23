# CodeSentinel-LLM

End-to-end Python bug localization and patch generation using
parameter-efficient fine-tuning (LoRA / QLoRA / DPO).

Given a Python code snippet containing a bug, the system:
1. **Localizes** the buggy line(s) using a GraphCodeBERT-based classifier
2. **Generates** a candidate fix using a fine-tuned Qwen2.5-Coder-7B model
3. **Splices** the fix back into the original code

## Results

| Model | Exact Match | Char-Similarity | High-Sim ≥ 0.8 |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (zero-shot baseline) | 0.066 | 0.593 | 0.331 |
| + SFT fine-tuning | 0.225 | 0.859 | 0.731 |
| + DPO refinement | 0.220 | 0.864 | 0.745 |

Localizer: Top-1 accuracy 0.806, Top-3 0.971, MRR 0.888

## Setup

```bash
pip install -r requirements.txt
```

## Download model weights

Weights are hosted on HuggingFace:

```python
from huggingface_hub import snapshot_download
snapshot_download("tripathiashutosh/codesentinel-localizer")
snapshot_download("tripathiashutosh/codesentinel-dpo")
```

Or update the paths in each script to point to your local checkpoints.

## Usage

**Desktop GUI** (paste code, select lines, click Repair):
```bash
python3 repair_gui.py
```

**Command-line pipeline** (detect with Semgrep + Bandit, auto-patch):
```bash
pip install semgrep bandit
python3 pipeline.py path/to/your_file.py
```

**Web GUI** (browser-based):
```bash
pip install streamlit
streamlit run gui.py
```

## Dataset

Trained on 31,709 Python bug-fix pairs from:
- [TSSB-3M](https://github.com/cedricrupb/TSSB-3M) — single-statement bug fixes
- [CVEfixes v1.0.8](https://github.com/secureIT-project/CVEfixes) — security patches

## Training

```bash
# Preprocess
python3 phase2_v2_preprocess.py

# Train localizer (GraphCodeBERT + LoRA)
python3 phase3_v3_fixed.py

# Train generator (Qwen2.5-Coder-7B + QLoRA)
python3 phase4a_sft.py

# DPO refinement
python3 phase4b_stage1_sample.py   # generate candidates
python3 phase4b_stage2_dpo.py      # DPO training
```

## Project report

Full technical report available in `report/report.pdf`.

## Author

Ashutosh Tripathi — B.Tech CSE, IIIT Raichur (2027)  
Under the supervision of Dr. Neha, Dept. of CSE, IIIT Raichur
