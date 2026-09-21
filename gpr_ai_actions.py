"""Azioni strutturate suggerite dall'AI: parsing, descrizione e applicazione al viewer."""

from __future__ import annotations

import json
import re

from .gpr_processing import FILTER_REGISTRY

# filtro -> {nome_parametro_AI: (attributo_widget_viewer, tipo)}
# tipo: int | float | bool | text (currentText) | data (currentData)
PARAM_MAP = {
    "dewow": {"window": ("_spin_dewow", "int")},
    "timezero": {
        "method": ("_cb_tz_method", "text"),
        "mode": ("_cb_tz_mode", "text"),
        "threshold": ("_spin_tz_threshold", "float"),
        "backup": ("_spin_tz_backup", "int"),
    },
    "bg_removal": {
        "method": ("_cb_bg_method", "data"),
        "mode": ("_cb_bg_mode", "data"),
        "auto": ("_chk_bg_auto", "bool"),
        "window": ("_spin_bg_window", "int"),
        "sample_start": ("_spin_bg_sample_start", "int"),
        "sample_end": ("_spin_bg_sample_end", "int"),
    },
    "bandpass": {
        "low_mhz": ("_spin_bp_lo", "float"),
        "high_mhz": ("_spin_bp_hi", "float"),
        "taper_pct": ("_spin_bp_taper", "float"),
    },
    "agc": {
        "window": ("_spin_agc", "int"),
        "max_gain": ("_spin_agc_max_gain", "float"),
        "sample_start": ("_spin_agc_sample_start", "int"),
    },
    "pre_agc_gain": {
        "gain_surface": ("_spin_gain_surface", "float"),
        "gain_deep": ("_spin_gain_deep", "float"),
    },
    "migration": {
        "velocity_m_s": ("_spin_mig_vel", "float"),
        "dx_m": ("_spin_mig_dx", "float"),
    },
}

ACTIONS_PROMPT = (
    "Quando consigli filtri o parametri, alla FINE della risposta aggiungi UN blocco ```json "
    "con questo schema (e nient'altro dentro il blocco):\n"
    '{"order": ["dewow","timezero",...], "actions": [{"filter": "<id>", "enable": true, '
    '"params": {"<nome>": <valore>}, "why": "motivo breve"}]}\n'
    "Id filtri ammessi: " + ", ".join(FILTER_REGISTRY) + ".\n"
    "Parametri ammessi per filtro: "
    + "; ".join(f"{f}: {', '.join(m)}" for f, m in PARAM_MAP.items())
    + ".\nValori: timezero.method in [peak, threshold, zero_crossing, dominant, surface]; "
    "timezero.mode in [line_by_line, scan_by_scan, individual]; "
    "bg_removal.method in [mean, median]; bg_removal.mode in [line_by_line, grid_by_grid]; "
    "bg_removal.auto true/false; frequenze in MHz; finestre e sample_start in campioni. "
    "Ometti il blocco json se non hai azioni da proporre."
)


def parse_actions(text: str) -> tuple[dict | None, str]:
    """Estrae il blocco JSON delle azioni. Ritorna (dict|None, testo senza blocco)."""
    if not text:
        return None, ""
    m = None
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S):
        pass
    if m is None:
        return None, text
    try:
        data = json.loads(m.group(1))
    except Exception:
        return None, text
    clean = (text[:m.start()] + text[m.end():]).strip()
    acts = []
    for a in data.get("actions", []) if isinstance(data, dict) else []:
        if not isinstance(a, dict) or a.get("filter") not in FILTER_REGISTRY:
            continue
        acts.append({
            "filter": a["filter"],
            "enable": bool(a.get("enable", True)),
            "params": a.get("params") if isinstance(a.get("params"), dict) else {},
            "why": str(a.get("why", "")),
        })
    order = [x for x in data.get("order", []) if x in FILTER_REGISTRY] if isinstance(data, dict) else []
    if not acts and not order:
        return None, clean
    return {"actions": acts, "order": order}, clean


def describe_action(a: dict) -> str:
    label = FILTER_REGISTRY.get(a["filter"], {}).get("label", a["filter"])
    state = "attiva" if a["enable"] else "disattiva"
    ps = ", ".join(f"{k}={v}" for k, v in a["params"].items())
    txt = f"{label}: {state}" + (f" [{ps}]" if ps else "")
    if a.get("why"):
        txt += f" - {a['why']}"
    return txt


def _set_widget(w, kind: str, value) -> None:
    if kind == "bool":
        w.setChecked(bool(value))
    elif kind == "int":
        w.setValue(int(round(float(value))))
    elif kind == "float":
        w.setValue(float(value))
    elif kind == "text":
        i = w.findText(str(value))
        if i < 0:
            raise ValueError(f"valore '{value}' non ammesso")
        w.setCurrentIndex(i)
    elif kind == "data":
        i = w.findData(str(value))
        if i < 0:
            raise ValueError(f"valore '{value}' non ammesso")
        w.setCurrentIndex(i)


def apply_actions(viewer, actions: list[dict], order: list[str] | None = None) -> list[str]:
    """Applica le azioni ai controlli del viewer e ricalcola. Ritorna un log leggibile."""
    log: list[str] = []
    chain = getattr(viewer, "_filter_chain_widget", None)
    enabled: dict[str, bool] = {}
    for a in actions:
        fid = a["filter"]
        for name, value in a["params"].items():
            spec = PARAM_MAP.get(fid, {}).get(name)
            if spec is None:
                log.append(f"  ! {fid}.{name}: parametro non supportato, ignorato")
                continue
            w = getattr(viewer, spec[0], None)
            if w is None:
                log.append(f"  ! {fid}.{name}: controllo non disponibile")
                continue
            try:
                _set_widget(w, spec[1], value)
            except Exception as exc:
                log.append(f"  ! {fid}.{name}={value}: {exc}")
        enabled[fid] = a["enable"]
        if fid == "migration" and getattr(viewer, "_chk_mig", None) is not None:
            viewer._chk_mig.setChecked(a["enable"])
        log.append(f"  ok {describe_action(a)}")
    if chain is not None:
        if order:
            rest = [s for s in chain.get_ordered_chain_ids() if s not in order]
            chain.set_order(list(order) + rest)
        chain.set_enabled_map(enabled, emit_signal=False)
        try:
            viewer._reveal_enabled_hidden_blocks()
        except Exception:
            pass
        viewer._on_filter_chain_widget_changed()
    else:
        viewer._apply_processing()
    return log
