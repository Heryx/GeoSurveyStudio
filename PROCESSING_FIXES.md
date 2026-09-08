# GPR Processing — Analisi bug e piano di fix

**Data analisi:** 2026-04-13  
**Branch di lavoro:** `claude/brave-mendeleev`  
**File principali:** `gpr_processing.py`, `gpr_profile_viewer.py`

---

## Problema riportato

> "Il gain continua ad essere applicato da 130 sample. Sembra cambiare molto se è attivo o meno il bandpass."

---

## Bug identificati

---

### Bug 1 — `normalize_display` annulla il `manual_gain` (radice del problema)

**File:** `gpr_profile_viewer.py` — `_compose_display_data()` (riga ~3377)

**Meccanismo:**

```python
# In _compose_display_data():
base = normalize_display(proc_disp)   # ← divide per il 98° percentile dell'INTERO array
gain = float(self._spin_gain.value())
disp = base * gain
```

`normalize_display` calcola il 98° percentile di tutti i valori assoluti
della matrice e divide tutto per quel valore. Se `manual_gain` ha amplificato
i campioni profondi 10–20×, il p98 cresce di conseguenza, e la divisione
annulla esattamente l'effetto del gain. Risultato visivo: il radargram
appare identico con gain ON o OFF.

**Fix:**  
`manual_gain` deve essere applicato **dopo** `normalize_display`, non prima.
La pipeline produce `proc_data` normalizzato a ±1. Il `manual_gain` poi
scala le righe profonde oltre 1.0, e `imshow(vmin=-1, vmax=1)` le satura
— esattamente l'effetto visivo desiderato (saturazione=bianco/nero pieno
in profondità, come fa ogni software GPR).

Struttura corretta in `_compose_display_data`:

```
proc_data  →  normalize_display  →  range_gain (display)  →  manual_gain  →  clip(±1)  →  imshow
```

Questo richiede di:
1. Togliere `manual_gain` dalla pipeline `apply_pipeline` (processing).
2. Applicare `manual_gain` come step display-only in `_compose_display_data`,
   dopo `normalize_display` e prima del `np.clip`.
3. In `apply_pipeline` rimuovere lo step `manual_gain` dalla chain.

---

### Bug 2 — `manual_gain` assente da `has_active_filters`

**File:** `gpr_profile_viewer.py` — `_apply_processing()` (riga ~2718)

```python
has_active_filters = any(
    bool(params.get(k, False))
    for k in (
        "timezero", "dewow", "bg_removal",
        "bandpass", "agc", "envelope", "pre_agc_gain",
    )
)
```

`"manual_gain"` manca. Se l'utente attiva solo `manual_gain` (e disattiva
tutti gli altri filtri) la pipeline non viene eseguita affatto:
`_proc_data = raw_data.copy()` e il gain non viene mai applicato.

Con il fix del Bug 1 questo diventa irrilevante per la pipeline, ma la
lista va comunque mantenuta corretta per futura robustezza.
→ **Fix:** aggiungere `"manual_gain"` alla lista.

---

### Bug 3 — Posizione di `manual_gain` nella chain (conseguenza del Bug 1)

**File:** `gpr_processing.py` — `DEFAULT_CHAIN_ORDER`

Il commit precedente ha spostato `manual_gain` PRIMA di `agc`, ma:
- Se gain prima di AGC: AGC normalizza l'energia locale → cancella il gain.
- Se gain dopo AGC: `normalize_display` in display cancella il gain.

Con il fix del Bug 1 (`manual_gain` → step display, fuori da pipeline):
`manual_gain` va rimosso completamente da `DEFAULT_CHAIN_ORDER` e dal
dispatcher `apply_pipeline`.

---

### Bug 4 — AGC cumsum: off-by-one nella finestra (minore, non causa il problema visivo)

**File:** `gpr_processing.py` — `agc_gain()` (riga ~427)

```python
sq_pad = np.pad(sq, ((half, half), (0, 0)), mode="reflect")
cs = np.vstack([np.zeros((1, n_t)), np.cumsum(sq_pad, axis=0)])
s_idx = np.arange(n_s)
sum_win = cs[s_idx + win, :] - cs[s_idx, :]
```

`s_idx` va da 0 a n_s-1. Per il campione `i`, la finestra copre
`sq_pad[i : i+win]`. L'origine `i=0` inizia al BORDO del padding,
non al centro del padding. La finestra di campioni originali va
`sq_pad[half : half+win]` per `i=0`, ma `cs[s_idx + win] - cs[s_idx]`
con `s_idx=0` calcola `cs[win] - cs[0]` = `sq_pad[0 : win]`.

Questo è CORRETTO: `sq_pad[0:win]` = riflesso(half) + originale(half)
→ finestra centrata sul campione 0 del dato originale. ✓

Nessun fix necessario qui. L'AGC è corretto.

---

### Bug 5 — Bandpass: DC removal per-traccia applicato prima del padding

**File:** `gpr_processing.py` — `_bandpass_fft()` (riga ~95)

```python
block = block - np.mean(block, axis=1, keepdims=True)
block_pad = np.pad(block, ((0, 0), (pad, pad)), mode="reflect")
```

Il DC removal viene applicato sul segnale NON paddato. Questo è corretto
(si rimuove la media prima del padding reflect per evitare discontinuità
al bordo). Non è la causa del problema visivo.

**Osservazione:** la sottrazione `block - mean` rimuove la componente
continua, il che modifica l'ampiezza relativa dei primi campioni rispetto
al segnale principale. Nessuna azione richiesta — comportamento atteso.

---

## Piano di fix (ordine esatto di applicazione)

### Passo 1 — `gpr_processing.py`

1. Rimuovere `manual_gain` da `DEFAULT_CHAIN_ORDER`.
2. Rimuovere il blocco `if step == "manual_gain":` da `apply_pipeline`.
3. Rimuovere le chiavi `"manual_gain"`, `"manual_gain_type"`,
   `"manual_gain_start"`, `"manual_gain_end"`, `"manual_gain_sample_start"`,
   `"manual_gain_sample_end"` da `DEFAULT_PIPELINE`.
4. Lasciare la funzione `manual_gain()` nel file — viene richiamata
   direttamente dal viewer come step display.

### Passo 2 — `gpr_profile_viewer.py`

**In `_compose_display_data()`:**

Sostituire la logica attuale con:

```python
def _compose_display_data(self):
    if self._proc_data is None:
        return None
    proc = np.asarray(self._proc_data, dtype=np.float32)
    if proc.ndim != 2 or proc.size <= 0:
        return None

    proc_disp = proc.astype(np.float32, copy=True)

    # 1. Range gain display (pre_agc_gain display-only se non già in pipeline)
    chk_rg_on = bool(... _chk_range_gain.isChecked() ...)
    last_pre_agc = bool(... _last_pipeline_params.get("pre_agc_gain") ...)
    if not (chk_rg_on and last_pre_agc):
        depth_gain = self._build_depth_range_gain(proc_disp.shape[0])
        if depth_gain is not None:
            proc_disp = (proc_disp * depth_gain).astype(np.float32)

    # 2. Normalizza PRIMA del manual_gain
    base = normalize_display(proc_disp)

    # 3. Scalare display gain
    gain = float(self._spin_gain.value())
    disp = base * gain

    # 4. Manual gain applicato DOPO normalize — l'imshow è clamped a ±1
    #    quindi l'eccesso saturerà (bianco/nero) come atteso
    if bool(... _chk_manual_gain.isChecked() ...):
        disp = _apply_manual_gain_display(disp, params_manual_gain)

    return np.clip(disp, -1.0, 1.0).astype(np.float32)
```

Importare `manual_gain` in cima al viewer (già importato? controllare).

**In `_apply_processing()` — `has_active_filters`:**

Aggiungere `"manual_gain"` alla lista dei filtri attivi (tecnica, per
completezza — con il nuovo schema non è più necessario ma è corretto).

**In `_current_processing_params()`:**

Rimuovere le 6 chiavi `manual_gain_*` dal dict restituito (non servono
più alla pipeline).

**In `FILTER_REGISTRY` e `_pipeline_chain_help_text`:**

Rimuovere la voce `"manual_gain"` perché non fa più parte della chain.
Nota: i widget `_chk_manual_gain`, `_spin_manual_gain_*` restano e
controllano lo step display. Aggiornare le label/tooltip di conseguenza.

**In `_build_pipeline_step_controls()`:**

Il blocco `if sid == "manual_gain"` non sarà più raggiunto dalla chain
widget, ma va lasciato inattivo (o rimosso) senza rompere nulla.

**In `_collect_gain_hyper_preset()` / `_apply_gain_hyper_preset()`:**

Le chiavi JSON del preset `manual_gain` vanno mantenute per compatibilità
con preset salvati in precedenza.

---

## Ordine di lavoro nella prossima sessione

1. Applicare Passo 1 a `gpr_processing.py` (rimozione da pipeline).
2. Applicare Passo 2 a `gpr_profile_viewer.py` (manual_gain come step display).
3. Committare su `claude/brave-mendeleev`.
4. Creare patch e applicare al branch principale
   `feature/interpolation-improvements-followup` (senza push su GitHub).
5. Verificare: con manual_gain ON e gain_end=20, i campioni profondi
   devono saturare a bianco/nero; i campioni superficiali restano a gain_start.
   L'effetto deve essere visibile indipendentemente da bandpass ON/OFF.

---

## Note tecniche aggiuntive

- `normalize_display` usa il 98° percentile come vmax. Dopo il manual_gain
  post-normalize, i valori >1.0 vengono clampati a ±1.0 da `np.clip`.
  Questo è il comportamento corretto: la saturazione in profondità è
  l'effetto visivo desiderato di un gain elevato.

- `imshow(vmin=-1, vmax=1)` in `_redraw()` già imposta i limiti corretti
  per visualizzare la saturazione. Nessun cambio necessario lì.

- Il `pre_agc_gain` (range gain con breakpoints) rimane in pipeline perché
  agisce PRIMA dell'AGC con l'obiettivo di attenuare/amplificare prima
  della normalizzazione. Ha scopo diverso dal `manual_gain` display.

- La funzione `manual_gain()` in `gpr_processing.py` resta perché è
  riutilizzata dal viewer come helper puro.
