# CodeSentinel-LLM

A desktop tool for automated Python bug repair using a fine-tuned large language model.

Paste your Python code, select the lines you think are buggy, and click **Repair**. The model generates a candidate fix, shows you a diff, and lets you accept or revert.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--7B-purple)
![Adapter](https://img.shields.io/badge/Fine--tuning-QLoRA%20%2B%20DPO-green)

---

## Demo

| Step | What happens |
|---|---|
| Paste code | Full Python code goes into the editor with line numbers |
| Select lines | Highlight 5–15 lines you suspect contain a bug |
| Click Repair | Model generates a fix, patched lines highlight green |
| Review diff | Red/green unified diff shown in the right panel |
| Accept or Revert | Keep the fix or undo it with one click |

---

## How it works

The repair model is a **Qwen2.5-Coder-7B** base model fine-tuned in two stages:

1. **Supervised fine-tuning (SFT)** on 31,709 Python bug-fix pairs from TSSB-3M and CVEfixes v1.0.8, using a localizer-window prompting strategy where the buggy region is marked with `# <BUG_START>` and `# <BUG_END>` sentinels
2. **Direct Preference Optimization (DPO)** using synthetically constructed preference pairs to further refine patch quality

Both stages use QLoRA (4-bit NF4 quantization + LoRA rank 16) so the full training and inference runs on a single consumer GPU.

### Results on held-out test set (3,112 records)

| Model | Exact Match | Char-Similarity | High-Sim ≥ 0.8 |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (zero-shot, no fine-tuning) | 0.066 | 0.593 | 0.331 |
| After SFT fine-tuning | 0.225 | 0.859 | 0.731 |
| After DPO refinement | 0.220 | 0.864 | 0.745 |

---

## Setup

**Requirements:** Python 3.10+, a CUDA-capable GPU with at least 8 GB VRAM

```bash
pip install -r requirements.txt
```

**Download the model adapter from HuggingFace:**

```python
from huggingface_hub import snapshot_download
snapshot_download("tripathiashutosh/codesentinel-dpo",
                  local_dir="codesentinel_data/phase4b_dpo/best")
```

Or set the `ADAPTER_DIR` and `BASE_MODEL` variables at the top of `repair_gui.py` to point to your local adapter path.

---

## Run

```bash
python3 repair_gui.py
```

A desktop window opens. No browser, no server, no extra setup.

The model loads in the background on startup (~30–60 seconds). The status bar in the top right shows **Model ready** in green when it is done.

---

## Usage

1. **Paste** your Python code into the editor (Ctrl+V)
2. **Select** the lines you suspect contain a bug by clicking and dragging
3. **Click Repair** — the button stays disabled until the model is ready and you have a selection
4. The patched lines **highlight green** and the diff panel on the right shows what changed
5. Click **✓ Accept** to keep the fix, or **✕ Revert** to go back to the original
6. Ctrl+Z works for normal typing undo at any point

---

## Important limitations

- The model works best on **5–15 line selections**. Very large selections may produce lower-quality patches.
- Generated patches should be **reviewed before use in production**, especially for security-sensitive code. The model can occasionally produce plausible-looking but incorrect fixes.
- The system operates on the selected region only. It does not analyze the rest of the file for context beyond the ±8 lines of surrounding code included in the prompt.

---

## Project report

Full technical report covering the dataset construction, model architecture, training procedure, evaluation results, and known limitations:

📄 [`report.pdf`](report/report.pdf)

---

## Author

**Ashutosh Tripathi**  
B.Tech Computer Science and Engineering  
Indian Institute of Information Technology, Raichur (2027)  
Under the supervision of **Dr. Neha**, Dept. of CSE, IIIT Raichur
