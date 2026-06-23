"""
CodeSentinel-LLM — Desktop Repair GUI
======================================
Paste Python code, select lines you suspect are buggy,
click Repair. The generator patches the selected region
in-place. No terminal commands needed.

Run:
    python3 repair_gui.py

Paths (edit if your project is elsewhere):
    ADAPTER_DIR   — LoRA adapter weights
    BASE_MODEL    — HuggingFace model id
"""

import os
import sys
import threading
import queue
import difflib
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk, font as tkfont, messagebox

# ─── CONFIG ────────────────────────────────────────────────
ADAPTER_DIR = Path("codesentinel_data/phase4b_dpo/best")
BASE_MODEL  = "Qwen/Qwen2.5-Coder-7B"

CONTEXT_LINES   = 8
MIN_WINDOW      = 6
MAX_WINDOW      = 80
MAX_NEW_TOKENS  = 256
BUG_START       = "# <BUG_START>"
BUG_END         = "# <BUG_END>"

# Colour palette — dark editor theme
BG_DARK     = "#1e1e2e"   # main background
BG_EDITOR   = "#11111b"   # code area
BG_PANEL    = "#181825"   # side panel
BG_GUTTER   = "#181825"   # line-number gutter
FG_TEXT     = "#cdd6f4"   # body text
FG_DIM      = "#6c7086"   # dim / placeholder
FG_GREEN    = "#a6e3a1"   # success
FG_RED      = "#f38ba8"   # error
FG_YELLOW   = "#f9e2af"   # warning
FG_BLUE     = "#89b4fa"   # accent / headings
FG_MAUVE    = "#cba6f7"   # highlight

ACCENT      = "#89b4fa"   # button accent
BTN_BG      = "#313244"   # button background
BTN_HOV     = "#45475a"   # button hover
SEL_BG      = "#45475a"   # editor selection

FONT_CODE   = ("JetBrains Mono", 11) if "JetBrains Mono" in tkfont.families() else \
              ("Courier New", 11)
FONT_UI     = ("Segoe UI", 10)       if sys.platform == "win32" else \
              ("SF Pro Display", 10) if sys.platform == "darwin" else \
              ("Ubuntu", 10)
FONT_BOLD   = FONT_UI + ("bold",)


# ─── PROMPT TEMPLATE ───────────────────────────────────────
PROMPT_TMPL = (
    "### Task:\n"
    "Fix the bug in the following Python code. The buggy region is marked "
    "with {bs} and {be}. Output ONLY the corrected version of that marked "
    "region — do not output the surrounding context.\n\n"
    "### Buggy code:\n```python\n{window}\n```\n\n"
    "### Fixed region:\n```python\n"
)


# ─── MODEL LOADER (runs once in a background thread) ───────
_model_ref  = [None]   # [model, tokenizer] once loaded
_load_q     = queue.Queue()

def _bg_load_model():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tok = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant,
            device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model.eval()
        _model_ref[0] = (model, tok)
        _load_q.put(("ok", None))
    except Exception as e:
        _load_q.put(("error", str(e)))


# ─── WINDOW EXTRACTION ─────────────────────────────────────
def build_window(code: str, lo_line: int, hi_line: int):
    """
    Given full code and a 1-indexed selected line range [lo, hi],
    build the localizer-window prompt.
    Returns (window_with_sentinels, ctx_lo, ctx_hi) or None.
    """
    lines = code.splitlines()
    n = len(lines)
    if lo_line < 1 or hi_line > n or lo_line > hi_line:
        return None
    ctx_lo = max(1, lo_line - CONTEXT_LINES)
    ctx_hi = min(n, hi_line + CONTEXT_LINES)
    while (ctx_hi - ctx_lo + 1) < MIN_WINDOW and (ctx_lo > 1 or ctx_hi < n):
        if ctx_lo > 1:    ctx_lo -= 1
        if ctx_hi < n and (ctx_hi - ctx_lo + 1) < MIN_WINDOW: ctx_hi += 1
    if (ctx_hi - ctx_lo + 1) > MAX_WINDOW:
        excess   = (ctx_hi - ctx_lo + 1) - MAX_WINDOW
        trim_top = excess // 2
        trim_bot = excess - trim_top
        ctx_lo   = min(lo_line, ctx_lo + trim_top)
        ctx_hi   = max(hi_line, ctx_hi - trim_bot)
    parts = []
    for i in range(ctx_lo, ctx_hi + 1):
        if i == lo_line:  parts.append(BUG_START)
        parts.append(lines[i - 1])
        if i == hi_line:  parts.append(BUG_END)
    return "\n".join(parts), ctx_lo, ctx_hi


def run_generator(window_text: str):
    """Run the loaded generator. Returns cleaned fix string."""
    import torch
    model, tok = _model_ref[0]
    device = next(model.parameters()).device
    prompt = PROMPT_TMPL.format(bs=BUG_START, be=BUG_END, window=window_text)
    ids = tok.encode(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
    fence = gen.find("```")
    if fence >= 0:
        gen = gen[:fence]
    return gen.rstrip()


# ─── MAIN APP ──────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CodeSentinel-LLM")
        self.configure(bg=BG_DARK)
        self.geometry("1100x750")
        self.minsize(800, 550)

        self._repair_queue = queue.Queue()
        self._loading      = True
        self._last_repair  = None    # (lo_line, hi_line, original_lines, fix)

        self._build_ui()
        self._start_loading()
        self._poll_queues()

    # ── BUILD UI ──────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Title bar
        hdr = tk.Frame(self, bg=BG_DARK, pady=10)
        hdr.grid(row=0, column=0, sticky="ew", padx=18)
        tk.Label(hdr, text="CodeSentinel", bg=BG_DARK, fg=FG_BLUE,
                 font=(FONT_CODE[0], 16, "bold")).pack(side="left")
        tk.Label(hdr, text=" — Bug Repair", bg=BG_DARK, fg=FG_DIM,
                 font=(FONT_UI[0], 12)).pack(side="left")
        self._status_var = tk.StringVar(value="Loading model…")
        tk.Label(hdr, textvariable=self._status_var, bg=BG_DARK, fg=FG_YELLOW,
                 font=FONT_UI).pack(side="right")

        # Body: editor on left, diff panel on right
        body = tk.Frame(self, bg=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_editor(body)
        self._build_panel(body)

        # Status / instructions bar
        foot = tk.Frame(self, bg=BG_PANEL, pady=6)
        foot.grid(row=2, column=0, sticky="ew")
        tk.Label(foot,
                 text="  Paste code → select 5-9 lines → Repair   "
                      "| Ctrl+Z to undo   | Accept / Revert after preview",
                 bg=BG_PANEL, fg=FG_DIM, font=(FONT_UI[0], 9)).pack(side="left")

    def _build_editor(self, parent):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text="Code editor", bg=BG_DARK, fg=FG_DIM,
                 font=(FONT_UI[0], 9)).grid(row=0, column=0, columnspan=3,
                                             sticky="w", pady=(0, 4))

        # Line-number gutter
        self._gutter = tk.Text(
            frame, width=4, bg=BG_GUTTER, fg=FG_DIM,
            font=FONT_CODE, state="disabled",
            relief="flat", bd=0, cursor="arrow",
            selectbackground=BG_GUTTER, selectforeground=FG_DIM,
        )
        self._gutter.grid(row=1, column=0, sticky="ns")

        # Main code text widget
        self._editor = tk.Text(
            frame, bg=BG_EDITOR, fg=FG_TEXT,
            insertbackground=FG_TEXT, selectbackground=SEL_BG,
            font=FONT_CODE, relief="flat", bd=0,
            undo=True, maxundo=50,
            wrap="none", padx=8, pady=6,
        )
        self._editor.grid(row=1, column=1, sticky="nsew")

        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical",
                             command=self._sync_scroll)
        vsb.grid(row=1, column=2, sticky="ns")
        self._editor.configure(yscrollcommand=lambda f, l: (
            vsb.set(f, l), self._update_gutter()))
        hsb = ttk.Scrollbar(frame, orient="horizontal",
                             command=self._editor.xview)
        hsb.grid(row=2, column=0, columnspan=3, sticky="ew")
        self._editor.configure(xscrollcommand=hsb.set)

        # Tags for highlighting
        self._editor.tag_configure("selection_hl",
                                    background="#2a2b3d", foreground=FG_TEXT)
        self._editor.tag_configure("repaired",
                                    background="#1e3a2f", foreground=FG_GREEN)

        # Toolbar below editor
        tb = tk.Frame(frame, bg=BG_DARK, pady=6)
        tb.grid(row=3, column=0, columnspan=3, sticky="ew")
        self._repair_btn = self._button(tb, "⚙  Repair selection",
                                         self._on_repair, ACCENT, state="disabled")
        self._repair_btn.pack(side="left", padx=(0, 8))
        self._button(tb, "Clear", self._on_clear, BTN_BG).pack(side="left")
        self._sel_label = tk.Label(tb, text="No selection", bg=BG_DARK,
                                    fg=FG_DIM, font=(FONT_UI[0], 9))
        self._sel_label.pack(side="right", padx=8)

        # Bind events
        self._editor.bind("<KeyRelease>", self._on_edit)
        self._editor.bind("<ButtonRelease-1>", self._on_edit)
        self._editor.bind("<B1-Motion>", self._on_edit)

    def _build_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        tk.Label(frame, text="Patch preview", bg=BG_PANEL, fg=FG_DIM,
                 font=(FONT_UI[0], 9), pady=8).grid(row=0, column=0,
                                                     sticky="w", padx=12)

        # Diff display
        self._diff_box = tk.Text(
            frame, bg=BG_PANEL, fg=FG_TEXT,
            font=FONT_CODE, relief="flat", bd=0,
            state="disabled", wrap="none", padx=8, pady=6,
        )
        self._diff_box.grid(row=2, column=0, sticky="nsew", padx=4)
        dvsb = ttk.Scrollbar(frame, orient="vertical",
                              command=self._diff_box.yview)
        dvsb.grid(row=2, column=1, sticky="ns")
        self._diff_box.configure(yscrollcommand=dvsb.set)

        self._diff_box.tag_configure("add",  foreground=FG_GREEN)
        self._diff_box.tag_configure("rem",  foreground=FG_RED)
        self._diff_box.tag_configure("hdr",  foreground=FG_MAUVE)
        self._diff_box.tag_configure("ctx",  foreground=FG_DIM)

        # Accept / Revert buttons
        btns = tk.Frame(frame, bg=BG_PANEL, pady=8)
        btns.grid(row=3, column=0, sticky="ew", padx=8)
        self._accept_btn = self._button(btns, "✓  Accept", self._on_accept,
                                         FG_GREEN, state="disabled",
                                         fg_text=BG_DARK)
        self._accept_btn.pack(side="left", padx=(0, 6))
        self._revert_btn = self._button(btns, "✕  Revert", self._on_revert,
                                         FG_RED, state="disabled",
                                         fg_text=BG_DARK)
        self._revert_btn.pack(side="left")

    # ── WIDGETS ───────────────────────────────────────────
    def _button(self, parent, text, cmd, bg, state="normal", fg_text=FG_TEXT):
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg_text, activebackground=BTN_HOV, activeforeground=FG_TEXT,
            font=FONT_BOLD, relief="flat", bd=0,
            padx=14, pady=6, cursor="hand2",
            state=state,
        )
        b.bind("<Enter>", lambda e: b.configure(bg=BTN_HOV))
        b.bind("<Leave>", lambda e: b.configure(bg=bg))
        return b

    # ── GUTTER ────────────────────────────────────────────
    def _update_gutter(self):
        self._gutter.configure(state="normal")
        self._gutter.delete("1.0", "end")
        lines = int(self._editor.index("end-1c").split(".")[0])
        nums  = "\n".join(str(i) for i in range(1, lines + 1))
        self._gutter.insert("1.0", nums)
        self._gutter.configure(state="disabled")
        # Sync gutter scroll position with editor
        self._gutter.yview_moveto(self._editor.yview()[0])

    def _sync_scroll(self, *args):
        self._editor.yview(*args)
        self._gutter.yview(*args)

    # ── EVENTS ────────────────────────────────────────────
    def _on_edit(self, event=None):
        self._update_gutter()
        self._update_sel_label()

    def _update_sel_label(self):
        try:
            sel_start = self._editor.index("sel.first")
            sel_end   = self._editor.index("sel.last")
            lo = int(sel_start.split(".")[0])
            hi = int(sel_end.split(".")[0])
            # If selection ends at column 0 of a line, that line isn't included
            if sel_end.split(".")[1] == "0" and hi > lo:
                hi -= 1
            n_lines = hi - lo + 1
            self._sel_label.configure(
                text=f"Lines {lo}–{hi}  ({n_lines} lines)",
                fg=FG_GREEN if 5 <= n_lines <= 15 else FG_YELLOW,
            )
            if _model_ref[0] is not None:
                self._repair_btn.configure(state="normal")
        except tk.TclError:
            self._sel_label.configure(text="No selection", fg=FG_DIM)
            self._repair_btn.configure(state="disabled" if _model_ref[0] is None
                                        else "disabled")

    # ── LOADING ───────────────────────────────────────────
    def _start_loading(self):
        t = threading.Thread(target=_bg_load_model, daemon=True)
        t.start()

    def _poll_queues(self):
        # Check model load queue
        try:
            status, msg = _load_q.get_nowait()
            if status == "ok":
                self._loading = False
                self._status_var.set("Model ready")
                self._status_label_color(FG_GREEN)
                self._update_sel_label()
            else:
                self._status_var.set(f"Load error: {msg[:60]}")
                self._status_label_color(FG_RED)
        except queue.Empty:
            pass

        # Check repair result queue
        try:
            rtype, payload = self._repair_queue.get_nowait()
            if rtype == "done":
                self._on_repair_done(payload)
            elif rtype == "error":
                self._status_var.set(f"Error: {payload[:80]}")
                self._status_label_color(FG_RED)
                self._repair_btn.configure(state="normal")
        except queue.Empty:
            pass

        self.after(120, self._poll_queues)

    def _status_label_color(self, color):
        # Find the status label — it's in the header
        for widget in self.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, tk.Label) and \
                   child.cget("textvariable") == str(self._status_var):
                    child.configure(fg=color)

    # ── REPAIR FLOW ───────────────────────────────────────
    def _on_repair(self):
        if _model_ref[0] is None:
            messagebox.showinfo("Not ready", "Model is still loading. Please wait.")
            return

        # Get selection
        try:
            sel_start = self._editor.index("sel.first")
            sel_end   = self._editor.index("sel.last")
        except tk.TclError:
            messagebox.showwarning("No selection",
                                   "Please select the lines you want to repair first.")
            return

        lo = int(sel_start.split(".")[0])
        hi = int(sel_end.split(".")[0])
        if sel_end.split(".")[1] == "0" and hi > lo:
            hi -= 1

        n_lines = hi - lo + 1
        if n_lines < 2:
            messagebox.showwarning("Selection too small",
                                   "Please select at least 2 lines.")
            return
        if n_lines > 20:
            if not messagebox.askyesno("Large selection",
                                        f"You selected {n_lines} lines. "
                                        "The model works best on 5-15 lines. Continue?"):
                return

        code = self._editor.get("1.0", "end-1c")
        result = build_window(code, lo, hi)
        if result is None:
            messagebox.showerror("Error", "Could not build window for selected lines.")
            return
        window_text, ctx_lo, ctx_hi = result

        # Disable button, show status
        self._repair_btn.configure(state="disabled")
        self._status_var.set(f"Repairing lines {lo}–{hi}…")

        # Store context for apply step
        self._pending = {
            "lo": lo, "hi": hi,
            "ctx_lo": ctx_lo, "ctx_hi": ctx_hi,
            "window_text": window_text,
            "original_code": code,
        }

        # Run generator in background
        def bg():
            try:
                fix = run_generator(window_text)
                self._repair_queue.put(("done", fix))
            except Exception as e:
                self._repair_queue.put(("error", str(e)))
        threading.Thread(target=bg, daemon=True).start()

    def _on_repair_done(self, fix: str):
        p = self._pending
        original_lines = p["original_code"].splitlines()
        lo, hi = p["lo"], p["hi"]

        # Build patched code
        fix_lines = fix.splitlines()
        patched_lines = original_lines[:lo - 1] + fix_lines + original_lines[hi:]
        patched_code  = "\n".join(patched_lines)

        # Store for accept/revert
        self._last_repair = {
            "lo": lo, "hi": hi,
            "original": p["original_code"],
            "patched":  patched_code,
            "fix_lines": fix_lines,
        }

        # Show diff in panel
        self._show_diff(original_lines, lo, hi, fix_lines)

        # Apply patch visually in editor (highlighted)
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", patched_code)
        self._editor.tag_remove("repaired", "1.0", "end")
        tag_lo = f"{lo}.0"
        tag_hi = f"{lo + len(fix_lines) - 1}.end"
        self._editor.tag_add("repaired", tag_lo, tag_hi)
        self._editor.see(tag_lo)
        self._update_gutter()

        # Update UI
        n_orig = hi - lo + 1
        n_new  = len(fix_lines)
        self._status_var.set(
            f"Patch ready — {n_orig} line(s) → {n_new} line(s). Accept or Revert.")
        self._accept_btn.configure(state="normal")
        self._revert_btn.configure(state="normal")
        self._repair_btn.configure(state="normal")

    def _show_diff(self, original_lines, lo, hi, fix_lines):
        before = original_lines[lo - 1 : hi]
        after  = fix_lines

        diff = list(difflib.unified_diff(
            before, after,
            fromfile=f"original (lines {lo}-{hi})",
            tofile="repaired",
            lineterm="",
            n=2,
        ))

        self._diff_box.configure(state="normal")
        self._diff_box.delete("1.0", "end")
        if not diff:
            self._diff_box.insert("end", "(no change — generator produced identical output)")
            self._diff_box.configure(state="disabled")
            return

        for line in diff:
            if line.startswith("+++") or line.startswith("---"):
                self._diff_box.insert("end", line + "\n", "hdr")
            elif line.startswith("@@"):
                self._diff_box.insert("end", line + "\n", "hdr")
            elif line.startswith("+"):
                self._diff_box.insert("end", line + "\n", "add")
            elif line.startswith("-"):
                self._diff_box.insert("end", line + "\n", "rem")
            else:
                self._diff_box.insert("end", line + "\n", "ctx")

        self._diff_box.configure(state="disabled")

    # ── ACCEPT / REVERT ───────────────────────────────────
    def _on_accept(self):
        if not self._last_repair:
            return
        # Remove highlight, keep patched code
        self._editor.tag_remove("repaired", "1.0", "end")
        self._status_var.set("Patch accepted.")
        self._accept_btn.configure(state="disabled")
        self._revert_btn.configure(state="disabled")
        self._last_repair = None
        # Clear diff panel
        self._diff_box.configure(state="normal")
        self._diff_box.delete("1.0", "end")
        self._diff_box.insert("end", "Patch accepted.")
        self._diff_box.configure(state="disabled")

    def _on_revert(self):
        if not self._last_repair:
            return
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", self._last_repair["original"])
        self._update_gutter()
        self._status_var.set("Reverted to original.")
        self._accept_btn.configure(state="disabled")
        self._revert_btn.configure(state="disabled")
        self._last_repair = None
        self._diff_box.configure(state="normal")
        self._diff_box.delete("1.0", "end")
        self._diff_box.insert("end", "Reverted.")
        self._diff_box.configure(state="disabled")

    # ── CLEAR ─────────────────────────────────────────────
    def _on_clear(self):
        self._editor.delete("1.0", "end")
        self._update_gutter()
        self._diff_box.configure(state="normal")
        self._diff_box.delete("1.0", "end")
        self._diff_box.configure(state="disabled")
        self._last_repair = None
        self._accept_btn.configure(state="disabled")
        self._revert_btn.configure(state="disabled")
        self._status_var.set("Ready" if not self._loading else "Loading model…")


# ── ENTRY POINT ────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
