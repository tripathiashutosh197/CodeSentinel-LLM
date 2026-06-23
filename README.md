# CodeSentinel-LLM

A desktop tool for automated Python bug repair using fine-tuned large language models.

Paste your Python code, select the lines you think are buggy, and click **Repair**. The system localizes the bug to the exact line, generates a candidate fix, shows you a colored diff, and lets you accept or revert — all in a native desktop window.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Localizer](https://img.shields.io/badge/Localizer-GraphCodeBERT%20%2B%20LoRA-orange)
![Generator](https://img.shields.io/badge/Generator-Qwen2.5--Coder--7B%20%2B%20QLoRA-purple)
![DPO](https://img.shields.io/badge/Refined%20with-DPO-green)

---

## How it works

The repair pipeline has two trained components working in sequence.

### 1. Bug Localizer — GraphCodeBERT + LoRA

A **GraphCodeBERT-base** (125M parameters) encoder fine-tuned with LoRA adapters (rank 16) to identify which lines within a code snippet are buggy. It operates as a per-token binary classifier; token-level probabilities are aggregated to line-level scores to produce a ranked list of candidate buggy lines.

**Localizer results on held-out test set (3,101 buggy samples):**

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.806 |
| Top-3 accuracy | 0.971 |
| Top-5 accuracy | 0.986 |
| Mean Reciprocal Rank (MRR) | **0.888** |

Top-3 accuracy of 0.971 means the correct buggy line is in the top 3 predictions 97.1% of the time.

### 2. Patch Generator — Qwen2.5-Coder-7B + QLoRA + DPO

A **Qwen2.5-Coder-7B** base model fine-tuned in two stages:

**Stage 1 — Supervised Fine-Tuning (SFT)** on 31,709 Python bug-fix pairs from TSSB-3M and CVEfixes v1.0.8. The model is trained with a localizer-window prompting strategy: the buggy region is wrapped with `# <BUG_START>` and `# <BUG_END>` sentinels, and the model is trained to emit only the corrected region.

**Stage 2 — Direct Preference Optimization (DPO)** using synthetically constructed preference pairs. High-temperature sampling produces candidate patches; the worst candidates (character similarity below 0.70 to gold) are used as rejected responses. This further refines the policy toward maintainer-style fixes.

Both stages use QLoRA (4-bit NF4 quantization + LoRA rank 16, 40.4M trainable parameters out of 4.39B total).

**Generator results on held-out test set (3,112 records):**

| Model | Exact Match | Char-Similarity | High-Sim >= 0.8 | Token-F1 |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (zero-shot, no fine-tuning) | 0.066 | 0.593 | 0.331 | 0.620 |
| After SFT fine-tuning | 0.225 | 0.859 | 0.731 | 0.817 |
| After DPO refinement | 0.220 | 0.864 | 0.745 | 0.823 |

Character similarity measures how close the generated patch is to the maintainer-written reference fix, on a scale from 0 (completely different) to 1 (identical).

### Dataset

| Source | Records | Description |
|---|---|---|
| TSSB-3M | 29,906 | Single-statement Python bug fixes mined from GitHub commits |
| CVEfixes v1.0.8 | 2,067 | Security vulnerability patches indexed by CVE and CWE |
| **Total** | **31,709** | After deduplication, stratified 80/10/10 train/val/test split |

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU with at least 8 GB VRAM

```bash
pip install -r requirements.txt
```

### Download model weights from HuggingFace

Two adapters are needed — the localizer and the generator:

```python
from huggingface_hub import snapshot_download

# 1. Bug localizer (GraphCodeBERT + LoRA)
snapshot_download(
    "cv-tihan/codesentinel-localizer",
    local_dir="codesentinel_data/phase3_v3_fixed/best_model"
)

# 2. Patch generator — DPO checkpoint (recommended)
snapshot_download(
    "cv-tihan/codesentinel-dpo",
    local_dir="codesentinel_data/phase4b_dpo/best"
)
```

The base model (`Qwen/Qwen2.5-Coder-7B`) loads automatically from HuggingFace the first time you run the GUI — no separate download needed.

---

## Run

```bash
python3 repair_gui.py
```

The window opens immediately. The model loads in the background — the status bar shows **Loading model...** in yellow, then **Model ready** in green when done (~30-60 seconds). The Repair button stays disabled until loading is complete.

---

## Usage

1. **Paste** your Python code into the editor
2. **Select** the lines you suspect contain a bug by clicking and dragging
3. **Click Repair** — runs in background, UI stays responsive
4. Patched lines **highlight green**, diff appears on the right panel
5. Click **Accept** to keep the fix or **Revert** to undo it

---

## Limitations

- Works best on **5-15 line selections**. The model was trained on snippets of 5-9 lines; very large selections may produce lower-quality patches.
- Generated patches should be **reviewed before production use**, particularly for security-sensitive code. The model can occasionally produce plausible-looking but incorrect fixes.
- The system operates on the selected region plus **+-8 lines of surrounding context** only. It does not analyze the full file.
- Whole-file bug detection is not built in. For automatic detection on a full file, use a static analyzer (Semgrep, Bandit) to identify candidate lines first.

---

## Technical report

Full technical report covering dataset construction, model architecture, training procedure, all evaluation results, diagnostic findings, and known limitations:

[report/report.pdf](report/report.pdf)

---

## Repository structure

```
CodeSentinel-LLM/
├── repair_gui.py        # Desktop GUI — main entry point
├── requirements.txt     # Python dependencies
├── report/
│   └── report.pdf       # Full technical report
└── README.md
```

Model weights are hosted on HuggingFace (see Setup above).

---

## Author

**Ashutosh Tripathi**
B.Tech Computer Science and Engineering
Indian Institute of Information Technology, Raichur (2027)
Under the supervision of **Dr. Neha**, Dept. of CSE, IIIT Raichur
