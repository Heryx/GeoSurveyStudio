"""Pannello 'Assistente AI': chat multi-turno, allegati, azioni applicabili ai filtri."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import tempfile
import urllib.request

import numpy as np
from qgis.PyQt.QtCore import QSettings, QThread, Qt, pyqtSignal
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

try:
    from qgis.PyQt.QtWidgets import QShortcut
except ImportError:  # Qt6
    from qgis.PyQt.QtGui import QShortcut

from .gpr_ai_actions import ACTIONS_PROMPT, apply_actions, describe_action, parse_actions
from .gpr_processing import LOG_FILE_PATH

PROVIDERS = {
    "Google Gemini": {"model": "gemini-2.5-flash", "hint": "API key da aistudio.google.com/apikey"},
    "Anthropic Claude": {"model": "claude-sonnet-5", "hint": "API key 'sk-ant-...' da console.anthropic.com"},
}

_SYSTEM = (
    "Sei un esperto di elaborazione GPR (georadar) per archeologia. "
    "Rispondi in italiano, in modo breve e concreto: diagnosi dei problemi, "
    "sequenza di filtri consigliata e parametri numerici da impostare.\n" + ACTIONS_PROMPT
)

QUICK_PROMPTS = [
    ("Consiglia i filtri", "Analizza il radargramma e consiglia sequenza e parametri dei filtri."),
    ("Time-zero corretto?", "Il time-zero e la rimozione dell'aria sopra il suolo sono corretti? Cosa cambieresti?"),
    ("Perche' niente iperboli?", "Perche' non vedo iperboli chiare? Cosa posso migliorare nell'elaborazione?"),
    ("Analizza il log", "Analizza il log e segnala anomalie nella pipeline."),
]

_TEXT_EXT = {".txt", ".log", ".csv", ".json", ".md", ".xml", ".py", ".ini", ".tsv"}
_MAX_TEXT = 30000
_MAX_BIN = 12 * 1024 * 1024


def read_log_tail(max_lines: int = 300) -> str:
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-max_lines:])
    except Exception as exc:
        return f"(log non leggibile: {exc})"


# ----------------------------------------------------------------------
# Richieste ai provider. messages: [{"role": "user"|"assistant", "parts": [...]}]
# part: {"text": str} | {"b64": str, "mime": str}
# ----------------------------------------------------------------------
def _build_request(provider, key, model, messages):
    if provider == "Google Gemini":
        contents = []
        for m in messages:
            gp = []
            for x in m["parts"]:
                if "text" in x:
                    gp.append({"text": x["text"]})
                else:
                    gp.append({"inline_data": {"mime_type": x["mime"], "data": x["b64"]}})
            contents.append({"role": "user" if m["role"] == "user" else "model", "parts": gp})
        body = {"system_instruction": {"parts": [{"text": _SYSTEM}]}, "contents": contents}
        return urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"content-type": "application/json", "x-goog-api-key": key},
        )
    msgs = []
    for m in messages:
        ap = []
        for x in m["parts"]:
            if "text" in x:
                ap.append({"type": "text", "text": x["text"]})
            elif x["mime"] == "application/pdf":
                ap.append({"type": "document", "source": {
                    "type": "base64", "media_type": x["mime"], "data": x["b64"]}})
            else:
                ap.append({"type": "image", "source": {
                    "type": "base64", "media_type": x["mime"], "data": x["b64"]}})
        msgs.append({"role": m["role"], "content": ap})
    body = {"model": model, "max_tokens": 2000, "system": _SYSTEM, "messages": msgs}
    return urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"},
    )


def _extract_text(provider, data):
    if provider == "Google Gemini":
        return "".join(
            pt.get("text", "")
            for c in data.get("candidates", [])
            for pt in c.get("content", {}).get("parts", [])
        )
    return "".join(b.get("text", "") for b in data.get("content", []))


class _AiWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, provider, key, model, messages, parent=None):
        super().__init__(parent)
        self._a = (provider, key, model, messages)

    def run(self):
        try:
            req = _build_request(*self._a)
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self.done.emit(_extract_text(self._a[0], data) or "(risposta vuota)")
        except Exception as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:
                pass
            self.failed.emit(f"{exc}\n{detail}")


class AiPanel(QWidget):
    def __init__(self, viewer, get_report, parent=None):
        super().__init__(parent)
        self._v = viewer
        self._get_report = get_report
        self._history: list[dict] = []
        self._attachments: list[dict] = []
        self._pending_actions: dict | None = None
        self._worker = None
        self._chat_md = ""
        st = QSettings()

        root = QVBoxLayout(self)

        # --- impostazioni provider (comprimibile) ---
        self._grp_set = QGroupBox("Provider e chiave")
        self._grp_set.setCheckable(True)
        sl = QHBoxLayout(self._grp_set)
        self._prov = QComboBox()
        self._prov.addItems(list(PROVIDERS))
        self._prov.setCurrentText(str(st.value("rasterlinker/ai_provider", "Google Gemini")))
        self._key = QLineEdit()
        self._key.setEchoMode(QLineEdit.Password)
        self._model = QLineEdit()
        for w, stretch in ((self._prov, 0), (self._key, 3), (self._model, 1)):
            sl.addWidget(w, stretch)
        self._prov.currentTextChanged.connect(self._load_provider_fields)
        self._load_provider_fields()
        self._grp_set.setChecked(not bool(self._key.text().strip()))
        self._grp_set.toggled.connect(lambda on: [w.setVisible(on) for w in (self._prov, self._key, self._model)])
        for w in (self._prov, self._key, self._model):
            w.setVisible(self._grp_set.isChecked())
        root.addWidget(self._grp_set)

        # --- richieste rapide ---
        qrow = QHBoxLayout()
        for label, prompt in QUICK_PROMPTS:
            b = QPushButton(label)
            b.clicked.connect(lambda _c=False, p=prompt: self._quick(p))
            qrow.addWidget(b)
        qrow.addStretch(1)
        root.addLayout(qrow)

        # --- chat + azioni ---
        split = QSplitter(Qt.Vertical)
        self._chat = QTextBrowser()
        self._chat.setOpenExternalLinks(True)
        self._chat.setPlaceholderText(
            "Scrivi una domanda qui sotto o usa una richiesta rapida.\n"
            "L'AI vede parametri, diagnosi e (se spuntato) log, immagine e dati; "
            "puoi allegare file e immagini."
        )
        split.addWidget(self._chat)

        self._grp_act = QGroupBox("Azioni proposte dall'AI")
        al = QVBoxLayout(self._grp_act)
        self._act_list = QListWidget()
        al.addWidget(self._act_list)
        arow = QHBoxLayout()
        b = QPushButton("Applica selezionate")
        b.clicked.connect(self._apply_selected)
        arow.addWidget(b)
        self._chk_auto = QCheckBox("Applica automaticamente i consigli dell'AI")
        self._chk_auto.setChecked(str(st.value("rasterlinker/ai_auto_apply", "false")).lower() == "true")
        self._chk_auto.toggled.connect(lambda on: QSettings().setValue("rasterlinker/ai_auto_apply", bool(on)))
        arow.addWidget(self._chk_auto)
        arow.addStretch(1)
        al.addLayout(arow)
        self._grp_act.setVisible(False)
        split.addWidget(self._grp_act)
        split.setStretchFactor(0, 3)
        root.addWidget(split, 1)

        # --- allegati ---
        arow = QHBoxLayout()
        b = QPushButton("Allega file/immagini...")
        b.clicked.connect(self._attach_files)
        arow.addWidget(b)
        b = QPushButton("Incolla immagine")
        b.setToolTip("Allega l'immagine presente negli appunti (es. screenshot).")
        b.clicked.connect(self._paste_image)
        arow.addWidget(b)
        b = QPushButton("Rimuovi allegato")
        b.clicked.connect(self._remove_attachment)
        arow.addWidget(b)
        arow.addStretch(1)
        root.addLayout(arow)
        self._att_list = QListWidget()
        self._att_list.setMaximumHeight(54)
        root.addWidget(self._att_list)

        # --- contesto ---
        crow = QHBoxLayout()
        self._chk_ctx = QCheckBox("Includi contesto (parametri + diagnosi)")
        self._chk_ctx.setChecked(True)
        self._chk_log = QCheckBox("Log")
        self._chk_log.setChecked(True)
        self._chk_img = QCheckBox("Immagine radargramma")
        self._chk_img.setChecked(True)
        self._chk_data = QCheckBox("Dati grezzi (sintesi)")
        self._chk_data.setChecked(True)
        for c in (self._chk_ctx, self._chk_log, self._chk_img, self._chk_data):
            crow.addWidget(c)
        b = QPushButton("Anteprima invio")
        b.clicked.connect(self._preview_context)
        crow.addWidget(b)
        crow.addStretch(1)
        root.addLayout(crow)

        # --- prompt ---
        prow = QHBoxLayout()
        self._q = QPlainTextEdit()
        self._q.setPlaceholderText("Scrivi la tua domanda...  (Ctrl+Invio per inviare)")
        self._q.setFixedHeight(96)
        prow.addWidget(self._q, 1)
        col = QVBoxLayout()
        self._btn_ask = QPushButton("Invia")
        self._btn_ask.setMinimumHeight(40)
        self._btn_ask.clicked.connect(self.ask_ai)
        col.addWidget(self._btn_ask)
        b = QPushButton("Nuova chat")
        b.clicked.connect(self._new_chat)
        col.addWidget(b)
        prow.addLayout(col)
        root.addLayout(prow)
        sc = QShortcut(QKeySequence("Ctrl+Return"), self._q)
        sc.activated.connect(self.ask_ai)
        self._note = QLabel("Cliccando Invia, i dati spuntati e gli allegati vengono inviati al provider scelto.")
        self._note.setWordWrap(True)
        root.addWidget(self._note)

    # ------------------------------------------------------------------
    def _env_key(self, prov):
        names = ["GEMINI_API_KEY", "GOOGLE_API_KEY"] if prov == "Google Gemini" else ["ANTHROPIC_API_KEY"]
        for n in names:
            if os.environ.get(n):
                return os.environ[n]
        return ""

    def _load_provider_fields(self, *_):
        prov = self._prov.currentText()
        st = QSettings()
        self._key.setText(str(st.value(f"rasterlinker/ai_key/{prov}", "") or self._env_key(prov)))
        self._model.setText(str(st.value(f"rasterlinker/ai_model/{prov}", PROVIDERS[prov]["model"])))
        self._key.setPlaceholderText(PROVIDERS[prov]["hint"] + " (salvata in locale nelle impostazioni QGIS)")

    # ------------------------------------------------------------------
    def _attach_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Allega file", "",
            "Immagini, PDF e testo (*.png *.jpg *.jpeg *.webp *.gif *.pdf *.txt *.log *.csv *.json *.md);;Tutti (*.*)",
        )
        for p in paths:
            self._add_attachment(p)

    def _paste_image(self):
        img = QApplication.clipboard().image()
        if img is None or img.isNull():
            QMessageBox.information(self, "Allegati", "Negli appunti non c'e' nessuna immagine.")
            return
        fd, path = tempfile.mkstemp(prefix="rl_clip_", suffix=".png")
        os.close(fd)
        img.save(path, "PNG")
        self._add_attachment(path)

    def _add_attachment(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        mime = mimetypes.guess_type(path)[0] or ""
        try:
            size = os.path.getsize(path)
            if ext in _TEXT_EXT or mime.startswith("text/"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read(_MAX_TEXT + 1)
                if len(txt) > _MAX_TEXT:
                    txt = txt[:_MAX_TEXT] + "\n[... troncato ...]"
                att = {"name": os.path.basename(path), "kind": "text", "text": txt}
            elif mime.startswith("image/") or mime == "application/pdf":
                if size > _MAX_BIN:
                    raise ValueError("file troppo grande (max 12 MB)")
                with open(path, "rb") as f:
                    att = {"name": os.path.basename(path), "kind": "bin", "mime": mime,
                           "b64": base64.b64encode(f.read()).decode("ascii")}
            else:
                raise ValueError("tipo di file non supportato")
        except Exception as exc:
            QMessageBox.warning(self, "Allegati", f"{os.path.basename(path)}: {exc}")
            return
        self._attachments.append(att)
        self._att_list.addItem(QListWidgetItem(f"{att['name']}  ({att['kind']})"))

    def _remove_attachment(self):
        r = self._att_list.currentRow()
        if r >= 0:
            self._att_list.takeItem(r)
            self._attachments.pop(r)

    # ------------------------------------------------------------------
    def _dt_ns(self) -> float:
        try:
            return float(self._v._profiles[self._v._prof_idx].dt_ns)
        except Exception:
            return 0.117

    def _image_png_b64(self):
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            arr = self._v._proc_data if self._v._proc_data is not None else self._v._raw_data
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=np.float32)
            c = float(np.percentile(np.abs(arr), 98)) or 1.0
            fig = Figure(figsize=(9, 5), dpi=80)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.imshow(arr, cmap="gray", aspect="auto", vmin=-c, vmax=c)
            ax.set_xlabel("traccia")
            ax.set_ylabel("campione")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None

    def _data_summary(self) -> str:
        raw = self._v._raw_data
        if raw is None:
            return ""
        a = np.abs(np.asarray(raw, dtype=np.float64))
        blocks = [int(round(a[i:i + 8].mean())) for i in range(0, min(a.shape[0], 200), 8)]
        first = np.asarray(raw[:24, 0]).astype(int).tolist()
        return (
            f"shape (campioni, tracce) = {raw.shape}, dt_ns = {self._dt_ns():.4f}\n"
            f"ampiezza media |x| per blocchi di 8 campioni (primi 200): {blocks}\n"
            f"prima traccia, campioni 0-23: {first}\n"
            f"min/max = {float(raw.min()):.0f} / {float(raw.max()):.0f}"
        )

    def _context_text(self) -> str:
        parts = ["Contesto: profilo GPR (formato .ogpr) in elaborazione in QGIS."]
        if self._chk_ctx.isChecked():
            rep = self._get_report() or {}
            parts.append("Diagnosi automatica:\n" + "\n".join(rep.get("notes", [])))
            try:
                params = {k: v for k, v in self._v._current_processing_params().items()
                          if not isinstance(v, (list, dict))}
                parts.append("Parametri correnti:\n" + json.dumps(params, default=str))
            except Exception:
                pass
        if self._chk_data.isChecked():
            parts.append("Dati grezzi (sintesi):\n" + self._data_summary())
        if self._chk_log.isChecked():
            parts.append("Ultime righe del log:\n" + read_log_tail(80))
        return "\n\n".join(parts)

    def _preview_context(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Anteprima di cio' che viene inviato")
        dlg.resize(700, 500)
        lay = QVBoxLayout(dlg)
        te = QPlainTextEdit()
        te.setReadOnly(True)
        extra = []
        if self._chk_img.isChecked():
            extra.append("[immagine del radargramma corrente]")
        extra += [f"[allegato: {a['name']}]" for a in self._attachments]
        te.setPlainText(self._context_text() + ("\n\n" + "\n".join(extra) if extra else "")
                        + "\n\nDomanda: " + self._q.toPlainText().strip())
        lay.addWidget(te)
        dlg.exec()

    # ------------------------------------------------------------------
    def _quick(self, prompt: str):
        self._q.setPlainText(prompt)
        self.ask_ai()

    def _new_chat(self):
        self._history.clear()
        self._chat_md = ""
        self._chat.setMarkdown("")
        self._grp_act.setVisible(False)
        self._pending_actions = None

    def _render_chat(self):
        self._chat.setMarkdown(self._chat_md)
        sb = self._chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    def ask_ai(self):
        if self._worker is not None and self._worker.isRunning():
            return
        question = self._q.toPlainText().strip()
        if not question:
            return
        key = self._key.text().strip().strip("\"'").strip()
        if not key:
            self._grp_set.setChecked(True)
            QMessageBox.information(self, "AI", "Inserisci la API key.")
            return
        prov = self._prov.currentText()
        model = self._model.text().strip() or PROVIDERS[prov]["model"]
        st = QSettings()
        st.setValue("rasterlinker/ai_provider", prov)
        st.setValue(f"rasterlinker/ai_key/{prov}", key)
        st.setValue(f"rasterlinker/ai_model/{prov}", model)

        parts: list = []
        for a in self._attachments:
            if a["kind"] == "bin":
                parts.append({"b64": a["b64"], "mime": a["mime"]})
            else:
                parts.append({"text": f"[File allegato: {a['name']}]\n{a['text']}"})
        if self._chk_img.isChecked():
            b64 = self._image_png_b64()
            if b64:
                parts.append({"b64": b64, "mime": "image/png"})
        parts.append({"text": self._context_text() + "\n\nDomanda: " + question})

        names = [a["name"] for a in self._attachments]
        stored = [{"text": (f"[allegati: {', '.join(names)}]\n" if names else "")
                   + self._context_text() + "\n\nDomanda: " + question}]
        messages = self._history + [{"role": "user", "parts": parts}]

        self._chat_md += f"\n\n**Tu:** {question}\n\n" + ("*(allegati: " + ", ".join(names) + ")*\n\n" if names else "")
        self._chat_md += "*...in attesa della risposta...*"
        self._render_chat()
        self._q.clear()
        self._attachments.clear()
        self._att_list.clear()
        self._btn_ask.setEnabled(False)
        self._pending_store = stored
        self._worker = _AiWorker(prov, key, model, messages, self)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _strip_wait(self):
        marker = "*...in attesa della risposta...*"
        if self._chat_md.endswith(marker):
            self._chat_md = self._chat_md[: -len(marker)]

    def _on_done(self, text: str):
        self._btn_ask.setEnabled(True)
        self._strip_wait()
        actions, clean = parse_actions(text)
        self._history.append({"role": "user", "parts": self._pending_store})
        self._history.append({"role": "assistant", "parts": [{"text": text}]})
        self._chat_md += f"**AI:**\n\n{clean}\n"
        self._render_chat()
        self._show_actions(actions)

    def _on_failed(self, msg: str):
        self._btn_ask.setEnabled(True)
        self._strip_wait()
        hint = ""
        if "401" in msg or "403" in msg or "API key" in msg or "API_KEY" in msg:
            hint = ("\n\nLa chiave non e' valida per il provider scelto: Gemini = aistudio.google.com/apikey; "
                    "Anthropic = 'sk-ant-...' da console.anthropic.com (l'abbonamento Claude.ai non e' una API key).")
        self._chat_md += f"**Errore:** {msg}{hint}\n"
        self._render_chat()

    # ------------------------------------------------------------------
    def _show_actions(self, actions):
        self._pending_actions = actions
        self._act_list.clear()
        if not actions:
            self._grp_act.setVisible(False)
            return
        if actions.get("order"):
            it = QListWidgetItem("Ordine catena: " + " > ".join(actions["order"]))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, "order")
            self._act_list.addItem(it)
        for i, a in enumerate(actions["actions"]):
            it = QListWidgetItem(describe_action(a))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, i)
            self._act_list.addItem(it)
        self._grp_act.setVisible(True)
        if self._chk_auto.isChecked():
            self._apply_selected(auto=True)

    def _apply_selected(self, auto=False):
        acts = self._pending_actions
        if not acts:
            return
        chosen, order = [], None
        for r in range(self._act_list.count()):
            it = self._act_list.item(r)
            if it.checkState() != Qt.Checked:
                continue
            tag = it.data(Qt.UserRole)
            if tag == "order":
                order = acts["order"]
            else:
                chosen.append(acts["actions"][int(tag)])
        if not chosen and not order:
            return
        log = apply_actions(self._v, chosen, order)
        head = "Applicati automaticamente" if auto else "Applicati"
        self._chat_md += f"\n*{head} {len(chosen)} consigli:*\n\n" + "\n".join(f"    {l}" for l in log) + "\n"
        self._render_chat()
