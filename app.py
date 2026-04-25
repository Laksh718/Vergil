"""
VERGIL — HuggingFace Spaces Training Monitor
=============================================
Serves a Gradio UI on port 7860 (required by HF Spaces).
Launches GRPO training in a background thread on boot.
Live-streams training logs. Pushes model to Hub when done.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Training state (shared between thread and Gradio) ──────────────────────
_logs: list[str] = []
_status: str = "Booting…"
_start_time: float | None = None
_done: bool = False


def _run_training() -> None:
    global _status, _done, _start_time
    _start_time = time.time()
    _status = "running"

    # Ensure we run from the repo root wherever HF mounts it
    cwd = Path(__file__).parent
    env = os.environ.copy()

    _logs.append("=" * 60)
    _logs.append("  VERGIL GRPO Training — starting now")
    _logs.append("  Model : Qwen2.5-0.5B  |  LoRA rank=64")
    _logs.append("  Target: Laksh718/vergil-commitment-engine")
    _logs.append("=" * 60)

    try:
        proc = subprocess.Popen(
            [sys.executable, "scripts/train_grpo_colab.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd),
            env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                _logs.append(line)
                # Keep log buffer bounded
                if len(_logs) > 500:
                    _logs.pop(0)

        proc.wait()
        if proc.returncode == 0:
            _status = "done"
            _logs.append("")
            _logs.append("✅ TRAINING COMPLETE — model pushed to Hub")
        else:
            _status = f"error:{proc.returncode}"
            _logs.append(f"❌ Process exited with code {proc.returncode}")
    except Exception as exc:
        _status = f"error:{exc}"
        _logs.append(f"❌ Launch failed: {exc}")
    finally:
        _done = True


# ── Start training immediately on Space boot ───────────────────────────────
_thread = threading.Thread(target=_run_training, daemon=True)
_thread.start()


# ── Gradio UI ──────────────────────────────────────────────────────────────
import gradio as gr


def get_status() -> str:
    elapsed = int(time.time() - _start_time) if _start_time else 0
    m, s = divmod(elapsed, 60)
    if _status == "running":
        return f"🔄  TRAINING  —  {m}m {s}s elapsed  |  {len(_logs)} log lines"
    elif _status == "done":
        return f"✅  COMPLETE  —  finished in {m}m {s}s"
    elif _status.startswith("error"):
        return f"❌  ERROR  —  {_status}"
    return f"⏳  {_status}"


def get_logs() -> str:
    return "\n".join(_logs[-80:]) if _logs else "Waiting for training to start…"


def get_metrics() -> str:
    vp = Path("/tmp/vergil_grpo_output/validation_log.json")
    if vp.exists():
        import json
        try:
            data = json.loads(vp.read_text())
            if data:
                rows = ["Step  |  Mean Reward  |  Fulfillment", "-" * 38]
                for entry in data:
                    rows.append(
                        f"{entry['step']:>4}  |  {entry['mean_reward']:>+10.3f}  |  "
                        f"{entry['mean_fulfillment']:>9.1%}"
                    )
                return "\n".join(rows)
        except Exception:
            pass
    return "No checkpoints yet — first one at step 20…"


css = """
body { background: #0a0e1a; }
.gr-button { background: #3b82f6 !important; }
"""

with gr.Blocks(title="VERGIL Training Monitor", css=css) as demo:
    gr.Markdown("""
# ⟁ VERGIL — GRPO Training Monitor

**Training `Qwen2.5-0.5B` to reason about Commitment Dependency Graphs**

The model generates **4 rollouts per CDG scenario** and uses group-relative advantage
to learn which commitment decisions maximise long-term trust × fulfillment.
When done, weights are auto-pushed to `Laksh718/vergil-commitment-engine`.
""")

    with gr.Row():
        status_box = gr.Textbox(
            label="Status", value="Initialising…",
            interactive=False, lines=1,
        )

    with gr.Row():
        metrics_box = gr.Textbox(
            label="Validation Checkpoints",
            value="Waiting…",
            interactive=False,
            lines=6,
        )

    log_box = gr.Textbox(
        label="Live Training Logs (last 80 lines)",
        value="Starting…",
        lines=28,
        interactive=False,
        max_lines=28,
    )

    gr.Markdown("""
---
### What each step does
1. Samples a CDG scenario from the curriculum (stage 1 → 4 difficulty ramp)
2. Generates **4 candidate responses** (each is a full `<think>…</think>` + JSON decision)
3. Evaluates all 4 against the VERGIL reward function (7 components: fulfillment, trust, proactive, accuracy, penalties)
4. Computes group-relative advantage: rewards each completion relative to the group mean
5. Backpropagates — the LoRA rank-64 adapters update their weights

**Target:** reward improving from ~0.1 (random) → ~0.6+ (strategic)
""")

    # Auto-refresh every 4 seconds
    demo.load(fn=get_status, outputs=status_box, every=4)
    demo.load(fn=get_logs, outputs=log_box, every=4)
    demo.load(fn=get_metrics, outputs=metrics_box, every=10)


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
)
