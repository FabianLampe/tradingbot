"""LLM-based post-mortem layer for failed trades.

For each losing trade, we feed:
  - The original recommendation (score, action, symbol, date)
  - SHAP attribution (which features drove the score)
  - The actual headlines + FinBERT scores
  - The macro snapshot at the time
  - The realised return

…to an LLM that writes a German-language analysis: *Why did this fail?
What signal was misleading? Is there a pattern with other recent fails?*

Two backends:
  - **claude** (default): Anthropic API, fast & high quality, costs ~$0.01/postmortem
  - **local**: HuggingFace transformers, runs on the user's GPU farm, no cost

Switch via env `LLM_BACKEND=claude|local` and `LLM_MODEL=...`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

POSTMORTEM_PROMPT_DE = """Du bist ein erfahrener Quant-Trader und analysierst einen Fehltrade.

Hier sind die Daten:

**Trade:** {action} {symbol} am {asof_date}, Score {score:+.2f}, Halteperiode {horizon_days} Tage
**Realisierte Rendite:** {realised_return:+.2%}

**Top-Features die zu diesem Score geführt haben (SHAP-Attribution):**
{top_features_str}

**News-Sentiment-Inputs:**
{sentiment_inputs_str}

**Makro-Kontext zum Trade-Zeitpunkt:**
{macro_str}

Schreibe in 3–5 Sätzen auf Deutsch:
1. Warum wurde dieser Trade empfohlen (Hauptsignal)?
2. Was ging schief — welches Feature hat in die Irre geführt?
3. Gibt es eine Lektion oder Heuristik die wir für Zukunft hinzufügen sollten?

Sei konkret und nenne Zahlen. Vermeide Floskeln wie "der Markt war schwierig"."""


@dataclass
class PostmortemRequest:
    symbol: str
    asof_date: str
    action: str
    score: float
    horizon_days: int
    realised_return: float
    top_features: list[dict]      # [{"feature": "...", "value": ..., "shap": ...}, ...]
    sentiment_inputs: dict
    macro_snapshot: dict


def _format_features(top_features: list[dict]) -> str:
    lines = []
    for f in top_features:
        sign = "+" if f["shap"] >= 0 else ""
        lines.append(f"  - {f['feature']}: Wert={f['value']:.3f}, SHAP={sign}{f['shap']:.3f}")
    return "\n".join(lines) if lines else "  (keine SHAP-Daten verfügbar)"


def _format_sentiment(inputs: dict) -> str:
    if not inputs:
        return "  (keine News am Trade-Tag)"
    parts = [
        f"  - Anzahl Artikel: {inputs.get('n_articles', 0)}",
        f"  - Mittlerer Sentiment-Score: {inputs.get('mean_score', 0):.2f}",
    ]
    headlines = inputs.get("top_headlines", [])
    if headlines:
        parts.append("  - Top-Headlines:")
        for h in headlines[:5]:
            parts.append(f"    • {h}")
    return "\n".join(parts)


def _format_macro(macro: dict) -> str:
    if not macro:
        return "  (keine Makrodaten)"
    return "\n".join(f"  - {k}: {v:.2f}" for k, v in macro.items() if v is not None)


def build_prompt(req: PostmortemRequest) -> str:
    return POSTMORTEM_PROMPT_DE.format(
        action=req.action,
        symbol=req.symbol,
        asof_date=req.asof_date,
        score=req.score,
        horizon_days=req.horizon_days,
        realised_return=req.realised_return,
        top_features_str=_format_features(req.top_features),
        sentiment_inputs_str=_format_sentiment(req.sentiment_inputs),
        macro_str=_format_macro(req.macro_snapshot),
    )


# ---------- backends ----------

def _generate_claude(prompt: str, model: str) -> str:
    import anthropic   # type: ignore
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _generate_local(prompt: str, model: str) -> str:
    """Run a local HuggingFace causal LM. Heavy — only for offline analysis."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16,
                                                device_map="auto")
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, max_new_tokens=600, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def write_postmortem(req: PostmortemRequest) -> tuple[str, str]:
    """Returns (analysis_text, model_id_used)."""
    backend = os.getenv("LLM_BACKEND", "claude").lower()
    if backend == "claude":
        model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        text = _generate_claude(build_prompt(req), model=model)
    elif backend == "local":
        model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        text = _generate_local(build_prompt(req), model=model)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}")
    return text, f"{backend}:{model}"
