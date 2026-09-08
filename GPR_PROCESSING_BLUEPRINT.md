# GPR Processing — Blueprint di riscrittura

**Data:** 2026-09-08
**Obiettivo:** ridisegnare il motore di processing GPR del plugin, con un
catalogo filtri completo, un'architettura non distruttiva a varianti e un
modello di navigazione multi-profilo. Tutto in Python (`numpy` / `scipy`),
senza dipendenze a runtime da R.

Riferimento algoritmico: **RGPR** (E. Huber, G. Hans),
<https://github.com/emanuelhuber/RGPR>, licenza `GPL (>= 2)` — le funzioni
vengono riscritte in Python. Le altre voci sono tecniche di processing GPR
standard in letteratura.

Documento di design: nessun codice è ancora stato scritto sulla base di questo
file. Va revisionato prima dell'implementazione.

---

## 0. Provenienza e licenza

| Fonte | Licenza | Uso |
|---|---|---|
| RGPR (E. Huber, G. Hans) — <https://github.com/emanuelhuber/RGPR> | `GPL (>= 2)` | riferimento algoritmico + modello dati; funzioni riscritte in Python |
| Questo plugin (`GeoSurvey Studio`) | `GPL-2.0-or-later` | destinazione |

Le licenze GPL-2+ sono compatibili: una funzione riscritta a partire da RGPR è
opera derivata GPL e può stare nel plugin. Requisiti:

1. Ogni funzione ispirata a RGPR riporta in docstring:
   `# Algoritmo adattato da RGPR (E. Huber, G. Hans), GPL (>= 2), file R/<nome>.R`.
2. File `CREDITS.md` nella root del plugin con citazione completa di RGPR (repo,
   autori, licenza) e del riferimento bibliografico (Huber & Hans, 2018, *RGPR —
   An open-source package to process and visualize GPR data*, EGU / GPR2018).
3. Le tecniche di processing non coperte da RGPR (whitening spettrale, filtri di
   linea sulle griglie, FFT 2D delle time-slice, correzione staggering, ecc.)
   sono metodi standard descritti in letteratura GPR/geofisica e vengono
   implementati da zero.

---

## 1. Stato attuale — cosa esiste già

Il plugin è più avanti di quanto sembri. Riutilizzare, non riscrivere:

| Componente esistente | File | Ruolo nel nuovo design |
|---|---|---|
| Pipeline filtri + `chain_order` + `_resolve_chain_order` | `gpr_processing.py` | base del motore step-by-step |
| `stage_cache` (cache per-step con `_step_fingerprint`) | `gpr_processing.py:1014` | diventa la cache del grafo di varianti |
| `FILTER_REGISTRY` / `DEFAULT_CHAIN_ORDER` / `DEFAULT_PIPELINE` | `gpr_processing.py:778` | si estende con i nuovi filtri |
| `background_removal` line/grid + mean/median + finestra sample | `gpr_processing.py:285` | già allineato a RGPR; solo da estendere |
| Widget catena riordinabile drag-and-drop | `filter_chain_widget.py` | UI del grafo/catena |
| Modello "variante" su disco (`variant.json`, `list_variants`) | `gpr_npz_io.py:186` | **embrione** del modello a varianti non distruttive |
| Preview asincrona | `gpr_profile_viewer.py:378` (`_PreviewSliceWorker`) | invariato |
| Picker bandpass sullo spettro | `gpr_profile_viewer.py:2695` (`_on_bp_hist_click`) | modello per gli altri picker |
| Editor curva TVG interattivo | `gpr_profile_viewer.py:109` (`_RangeGainCurveDialog`) | modello per il picker velocità |
| Dialog posizionamento (griglia, spacing, zigzag, invert, offset, origine da canvas) | `gpr_manual_positioning_dialog.py` | base del layer navigazione |

Limiti reali da superare:

- **Catena lineare, non grafo.** Una sola `_proc_data`; impossibile tenere N
  varianti nominate vive e usarne una come input di un'altra.
- **Varianti flat.** `variant.json` è uno snapshot dell'intera pipeline, non
  un singolo step con genitore.
- **Set di filtri fisso** (8 step).
- **Migrazione** solo Stolt F-K a velocità costante.
- **Nessun filtro nel dominio griglia/time-slice.**
- **Navigazione**: solo geometria manuale; niente modello marker, flag reverse
  persistente, staggering, survey ad angolo arbitrario, multi information-file.

---

## 2. Modello dati (da RGPR)

### 2.1 `GprProfile` (oggetto profilo)

Struttura logica — non necessariamente una nuova classe, può essere un dict
tipizzato / dataclass:

```
GprProfile
  data        : (n_samples, n_traces) float32      # ampiezze
  t           : (n_samples,) float64               # asse tempo [ns] (o profondità)
  x           : (n_traces,) float64                # ascissa lungo il profilo [m]
  coords      : (n_traces, 3) float64 | None       # E, N, Z per traccia (se GPS)
  dt_ns       : float
  antenna_mhz : float
  crs_epsg    : int | None
  markers     : list[Marker]                       # vedi §5
  proc        : list[ProcStep]                     # storico di processing
  meta        : dict                               # manufacturer, file sorgente, ...
```

`proc` è lo **storico** (come `proc(x)` in RGPR): ogni elemento descrive uno step
già applicato — `{"id": "dewow", "params": {...}, "ts": "...", "fingerprint": ...}`.
Serve a: ricostruire una variante, mostrare la history in UI, invalidare la cache.

### 2.2 `GprSurvey` (da `GPRsurvey` di RGPR)

Collezione di profili con geometria comune:

```
GprSurvey
  profiles      : list[GprProfile | ref]
  crs_epsg      : int | None
  geometry      : list[LineGeom]        # x0,y0,x1,y1 per linea (information file)
  intersections : list[...]             # incroci fra linee (per quality check)
  topo          : TopoModel | None
```

`geometry` è un **information file**: una riga per linea con estremi sul terreno
(`file, x0, y0, x1, y1`). È già in parte in `gpr_manual_positioning_dialog`.

---

## 3. Architettura non distruttiva a grafo di varianti

Unifica `stage_cache` + `filter_chain_widget` + modello `variant.json`.

### 3.1 Concetto

Ogni **variante** è un nodo:

```
Variant
  id          : str            # es. "bandpass", "mig_v08"
  parent_id   : str | None     # "raw" alla radice
  step        : str            # id nel FILTER_REGISTRY
  params      : dict
  scope       : "line" | "survey"     # per-linea vs statistica sull'intero dataset
  cache_dir   : str            # <processed_root>/<id>/  con .npz per profilo + variant.json
  status      : "clean" | "stale" | "computing"
```

Il grafo è un albero (in futuro DAG per il blending multi-frequenza). La radice
`raw` è il dato convertito, mai modificato.

### 3.2 Esecuzione

- `resolve(variant_id)` → risale i `parent_id` fino a `raw`, restituisce la
  catena di step.
- `materialize(variant_id)` → esegue gli step mancanti partendo dalla cache
  valida più profonda; riusa `_step_fingerprint` per decidere validità.
- La **preview** in RAM resta com'è (catena lineare degli step abilitati nel
  pannello) — è semplicemente la variante "di lavoro" non ancora committata.
- **Commit variante** → scrive `.npz` per profilo + `variant.json` con
  `parent`, `step`, `params`, `scope` (estendere `write_variant_manifest`).

### 3.3 UI

- `filter_chain_widget` mostra la catena della variante di lavoro (com'è ora).
- Nuovo pannello "Varianti": albero delle varianti committate; doppio click =
  carica come input; menu contestuale "nuova variante da qui".
- Il **gridding / time-slice** prende in input una variante selezionata invece
  del solo `_proc_data` corrente.

---

## 4. Catalogo filtri (dominio radargramma)

Legenda stato: **✅ esiste** · **🔩 harden** (esiste, va esteso) · **🆕 nuovo**.
`R/<file>.R` indica il file RGPR di riferimento quando presente.

### 4.1 Tempo / correzioni base

| Step | Stato | Firma proposta | Note algoritmo |
|---|---|---|---|
| `dcshift` (rimozione DC per traccia) | 🆕 | `dcshift(data, u=None)` | media dei campioni `u` (o tutti) sottratta per traccia. RGPR `dcshift`. |
| `dewow` | 🔩 | `dewow(data, window)` | esiste; aggiungere metodo `"gaussian"` oltre a running-mean. RGPR `dewow` (type `runmed`/`runmean`/`gaussian`). |
| `time_zero` | 🔩 | esiste (`time_zero_correction`) | aggiungere stima t0 per traccia con cross-correlation (RGPR `firstBreak` + `time0Cor`, metodi `"MER"`, `"threshold"`, `"coppens"`). |
| `static_shift` | 🆕 | `static_shift(data, dt_per_trace)` | shift verticale per traccia da un vettore (topo o drift). |

### 4.2 Rimozione clutter / rumore coerente

| Step | Stato | Firma | Note |
|---|---|---|---|
| `bg_removal` (mean/median trace) | 🔩 | esiste, completo (line/grid, finestra, mean/median, sliding) | già allineato a RGPR `traceStat`. Nessun lavoro urgente. |
| `eigen_filter` (SVD / eigenimage) | 🆕 | `eigen_filter(data, keep=(2,None))` | SVD della matrice; azzera le prime k componenti (clutter orizzontale) o tiene solo un range. RGPR `eigenFilter`. Ottimo BG removal "morbido". |
| `median_xy` | 🆕 | `median_filter2d(data, w_trace, w_sample)` | mediana 2D; de-spike. RGPR `filter2D(type="median")`. |
| `notch` | 🆕 | `notch(data, dt_ns, f0_mhz, q)` | riusa `_bandpass_fft`; maschera gaussiana inversa attorno a f0. |

### 4.3 Filtri in frequenza

| Step | Stato | Firma | Note |
|---|---|---|---|
| `bandpass` | 🔩 | esiste (`bandpass_filter` / `_bandpass_fft`) | aggiungere finestratura (cosine taper) e ordine; opzione Butterworth via `scipy.signal`. RGPR `fFilter(type="bandpass")`. |
| `spectral_whitening` | 🆕 | `whiten(data, dt_ns, f_lo, f_hi, gain)` | FFT → \|mag\|=1 nella banda, fase preservata, 0 fuori → IFFT → `/gain`. Metodo standard. |
| `fk_filter` | 🆕 | `fk_filter(data, dt_ns, dx_m, mask)` | FFT 2D (f, kx); `mask` = poligono/pennello in UI; IFFT. RGPR `fkFilter`. |
| `fk_dip_filter` | 🆕 | `fk_dip(data, ..., vmin, vmax)` | passa/ferma per pendenza apparente (velocità apparente) — rimozione ground-roll / riflessioni dirette. |

### 4.4 Gain

| Step | Stato | Firma | Note |
|---|---|---|---|
| `pre_agc_gain` (TVG curva) | ✅ | esiste (`_build_pre_agc_gain` + `_RangeGainCurveDialog`) | ok. |
| `agc` | 🔩 | esiste | aggiungere modo `"rms"` oltre a quello attuale; `max_gain`. RGPR `gain(type="agc")`. |
| `gain_power` / `gain_exp` | 🆕 | `gain(data, t, type, a, b, t0)` | `t^a` e `exp(a·t)` — leggi di gain classiche. RGPR `gain(type="power"/"exp")`. |
| `amplitude_equalization` | 🆕 | `amp_eq(data, win)` | gain per campione = risoluzione binaria / RMS su finestra mobile centrata. Metodo standard (equalizzazione RMS). |
| `scan_gain_norm` | 🆕 | `scan_gain_norm(data, amp_horizon)` | normalizza ogni traccia con `1/((a-amin)/(amax-amin))` da ampiezza di un orizzonte (compensazione accoppiamento antenna). |

### 4.5 Deconvoluzione

| Step | Stato | Firma | Note |
|---|---|---|---|
| `decon_spiking` | 🆕 | `decon_spiking(data, nf, mu)` | Wiener/Levinson, filtro predittivo lag=1. RGPR `deconv(type="spiking")`. |
| `decon_predictive` | 🆕 | `decon_predictive(data, nf, lag, mu)` | rimozione multiple. RGPR `deconv(type="mixed-phase"/"wavelet")`. |
| `decon_cepstrum` | 🆕 | `decon_cepstrum(data, gate)` | FFT → log\|·\| → IFFT → liftering. Metodo standard (deconvoluzione omomorfa). |

### 4.6 Migrazione e velocità

| Step | Stato | Firma | Note |
|---|---|---|---|
| `migration_stolt` | ✅ | esiste (`stolt_migration_2d`) | F-K, velocità costante. |
| `migration_kirchhoff` | 🆕 | `mig_kirchhoff(data, t, x, v, aperture_m, tr_offset=0)` | somma su iperbole con correzioni obliquità + spreading; offset Tx-Rx opzionale. RGPR `migration(type="kirchhoff")`. |
| `migration_kirchhoff_topo` | 🆕 | `... , topo` | migrazione da superficie topografica. RGPR `migration` con `@coord`. |
| `migration_v_profile` | 🆕 | `v_model="const"\|"block"\|"profile"` | velocità variabile con la profondità da punti (hyperbola fit). Metodo standard (migrazione a velocità variabile). |
| `velocity_from_hyperbola` | 🆕 (tool) | picker interattivo | fit iperbola `t² = t0² + (2x/v)²` col mouse su una traccia; restituisce v, ε, profondità. Alimenta `migration_v_profile` e `t→z`. RGPR `hyperbolaFit`. |
| `cmp_velocity` | 🆕 (avanzato) | `cmp_analysis(cmp_profile)` | semblance / NMO per acquisizioni CMP/WARR. RGPR `CMPAnalysis`. Fase 5. |
| `nmo_correction` | 🆕 | `nmo(data, t, offset, v)` | correzione normal-moveout. RGPR `NMOCor`. |

### 4.7 Inviluppo / attributi

| Step | Stato | Firma | Note |
|---|---|---|---|
| `envelope` (Hilbert) | ✅ | esiste (`hilbert_envelope`) | ok. |
| `instantaneous_phase` / `freq` | 🆕 | `inst_attr(data, kind)` | attributi complessi da segnale analitico. RGPR `Hilbert` / attributi. |
| `radar_threshold` | 🆕 | `threshold(data, neg, pos)` | azzera \|pulse\| sotto soglia (min/max indipendenti). Metodo standard. |

### 4.8 Multi-traccia / geometria dati

| Step | Stato | Firma | Note |
|---|---|---|---|
| `trace_average` / `stack` | 🆕 | `stack(data, n)` | media ogni n tracce. RGPR `traceAverage`. |
| `boxcar_2d` | 🆕 | `boxcar(data, w, h)` | media 2D non pesata. Metodo standard (lisciatura boxcar). |
| `resample_t` / `resample_x` | 🔩 | verificare cosa esiste già | ricampionamento in tempo o in distanza (fallout GPS). RGPR `regInterpPos` / `upsample`. |
| `n_segment_interp` | 🆕 | `interp_between_lines(survey, n)` | N interpolazioni lineari fra profili adiacenti (+ interpolazione nav per survey GPS/vector). Metodo standard. |
| `multifreq_blend` | 🆕 | `blend(profiles_lo, profiles_hi, k)` | somma pesata di canali a frequenza diversa dopo allineamento t0 + resample. Richiede DAG. Fase 5. |

---

## 5. Navigazione e posizionamento multi-profilo

### 5.1 Modello marker

```
Marker
  trace_idx : int        # scan a cui è ancorato
  pos_m     : float       # ascissa nota lungo il profilo [m], o
  coord     : (E,N,Z)     # per navigazione GPS
  kind      : "field" | "artificial" | "interval" | "gps"
```

Fra due marker consecutivi la posizione degli scan è **dead-reckoned**
(interpolazione lineare).

### 5.2 I 4 modi di assegnazione

| Modo | Funzione | Uso |
|---|---|---|
| `artificial_markers(profile, length_m)` | N marker equidistanti sul totale scan | ruota odometrica con slittamento |
| `field_markers(profile)` | legge tag inseriti in campo (header o file ASCII) | marker manuali del rilievo |
| `interval_markers(profile, scans_per_mark)` | marker a passo fisso di scan | ruota "ideale" |
| `gps_trace_markers(profile, gps_file)` | un marker per lettura GPS (col. scan sincronizzato) | rilievi GPS/total station |

Regola da replicare: **l'ultima assegnazione sovrascrive i marker precedenti**;
i marker vanno rigenerati quando si affetta una variante diversa.

### 5.3 Geometria del survey

Estendere `gpr_manual_positioning_dialog`:

- **information file** persistente: `[{file, x0, y0, x1, y1, axis, reversed}]` →
  già quasi tutto presente (`_on_axis_changed`, `_on_offset_changed`,
  `_on_reversed_changed`, `_apply_zigzag`).
- **reverse / zig-zag**: flag `reversed` persistente per linea. Due modalità:
  flip fisico del dato, oppure `x1 < x0` nella geometria senza toccare il dato.
- **staggering correction**: `stagger_m` globale + file per-linea
  `{line, stagger_m}` applicato agli xyz prima del gridding (anche differenziale,
  solo su alcune linee o su un range di ascisse).
- **survey ad angolo arbitrario**: linee con lunghezza frazionaria; marker
  arrotondati per eccesso a posizioni non intere.
- **multi information-file** per progetto (sub-griglie) + append cross-XY
  (`infox` + `infoy` → `infoxy`).

### 5.4 Griglie multiple

- `rotate_translate_grid(grid, angle, origin, mode)` — `mode` = "fix origin"
  (SW corner fisso) | "translate origin".
- `append_gridsets(grids)` — merge di gridset con stessa cell size.
- `mosaic_correct(grids, method)` — `method` = 0-mean grid | 0-mean line |
  edge matching | histogram matching.

---

## 6. Filtri dominio griglia / time-slice

Secondo motore, opera sui raster delle slice — percorso
`grid_creator.py` / `gpr_slice_generator.py`, non sui radargrammi.

| Filtro | Firma | Note |
|---|---|---|
| `grid_lowpass` | `lowpass(grid, k)` | boxcar 3×3 … 9×9. |
| `grid_laplacian` | `laplacian(grid)` | enfasi bordi. |
| `grid_hipass` | `hipass(grid, k)` | |
| `grid_bg_line` | `bg_line(grid, axis, length, lo_thr, hi_thr)` | background di linea verticale/orizzontale sulla griglia, con soglie percentuali per non spalmare i target. |
| `grid_fft2d` | `fft2d_filter(grid, mask)` | blanking di frequenze spaziali (solchi aratura ecc.). |
| `grid_blank` | `blank(grid, polygon)` | maschera disegnata a mano. |

---

## 7. Strumenti di parametrizzazione interattiva

| Strumento | Stato | Modello |
|---|---|---|
| Picker bandpass su spettro (click lo/hi) | ✅ | `_on_bp_hist_click` |
| Editor curva TVG | ✅ | `_RangeGainCurveDialog` |
| Fit iperbola → velocità/ε/profondità | ✅ (Fase 2) | overlay iperbole + apex da cursore + auto-fit RDP + offset Tx-Rx bistatico; readout v [m/s e cm/ns], εr, profondità target; "Usa velocita' iperbole" → migrazione |
| Bandpass taper/roll-off regolabile | ✅ (Fase 2) | `bp_taper_frac` in `_bandpass_fft` (default 0.12 = comportamento precedente) |
| Painting maschera FK / FFT2D | 🆕 | canvas con pennello a raggio variabile + cattura rettangolare |
| Editor marker su radargramma | 🔩 | sx = inserisci, dx = cancella; overlay "mostra marker" |

---

## 8. Roadmap per fasi

**Fase 1 — filtri "facili", nessun cambio architetturale. ✅ FATTA (2026-09).**
`dcshift`, `boxcar_smooth`, `notch_filter`, `spectral_whitening`,
`amplitude_equalization`, `radar_threshold`, `power_gain`.
Funzioni pure in `gpr_processing.py` + `FILTER_REGISTRY` + `DEFAULT_PIPELINE`
(tutti OFF) + `DEFAULT_CHAIN_ORDER` (15 step) + `_step_fingerprint` +
branch in `apply_pipeline`. UI: factory generica `_build_phase1_step_controls`
in `gpr_profile_viewer.py` (spec `_PHASE1_SPECS`), enable dalla casella del
blocco chain, valori da `_phase1_params()`, merge in `_current_processing_params`.
17 test in `test/test_gpr_processing_adv.py`. Nessuna regressione:
`DEFAULT_CHAIN_ORDER` nuovo == ordine vecchio con i 7 filtri OFF.

**Fase 2 — parametrizzazione interattiva. ✅ FATTA (2026-09).**
- Bandpass: `bp_taper_frac` (0..0.5) reale in `_bandpass_fft` / `bandpass_filter`,
  default 0.12 (= comportamento precedente). Wiring completo
  `DEFAULT_PIPELINE` / `_step_fingerprint` / `apply_pipeline` / controlli step.
- Fit iperbola: offset Tx-Rx bistatico (`_spin_hyperbola_txrx`) nel modello
  dell'overlay e nell'auto-fit; readout arricchito (v in m/s **e** cm/ns, εr,
  profondità target `d = v·t0/2`).

**Fase 3 — architettura a grafo di varianti. ◑ CORE FATTO (2026-09), UI rinviata.**
- `variant_graph.py` (nuovo, puro Python, no Qt): `VariantNode` +
  `VariantGraph` con `add_node` / `remove_subtree` / `ancestors` / `children` /
  `leaves` / `rename` / `set_params`; `resolve(node)` → `(params completi,
  [step ordinati])` partendo da `DEFAULT_PIPELINE`; `content_key(node)` (hash
  stabile per staleness); `materialize_arrays(node, raw_arrays, dt_ns, cache)`
  che gira `apply_pipeline` per array riusando `stage_cache` sul prefisso
  invariato; persistenza `variant_graph.json` (`save`/`load`/`from_json` con
  insert topologico); `from_legacy_manifest()` converte un vecchio
  `variant.json` piatto in catena lineare.
- 16 test in `test/test_variant_graph.py` (topologia, resolve, materialize ==
  `apply_pipeline` diretto, cache reuse, branching, round-trip, legacy).
- **Rinviato:** pannello "Varianti" in `gpr_profile_viewer.py` e "gridding da
  variante selezionata". Motivo: il percorso di gridding/slicing è quello che
  `TIMESLICE_REDESIGN_WORKPLAN.md` marca come da riscrivere — l'integrazione U
  va fatta DOPO le fasi A/B di quel workplan (che introducono `SliceSet` /
  `compute_slice`), così il pannello Varianti alimenta direttamente il nuovo
  motore invece di aggiungere accoppiamento al vecchio.

**Fase 4 — migrazione avanzata + eigen/median/FK. ✅ FATTA (2026-09).**
Funzioni pure numpy in `gpr_processing.py` (nessuna dip. scipy):
- `eigen_filter(data, drop, keep_last)` — SVD (`np.linalg.svd`), azzera le prime
  `drop` componenti (clutter orizzontale coerente); `keep_last` isola una banda.
- `median_filter2d(data, w_trace, w_sample)` — mediana 2D via
  `sliding_window_view` (+ fallback), de-spike.
- `fk_filter(data, dt_ns, dx_m, mask)` — primitiva: applica una maschera FK
  fornita dal chiamante (la UI di painting la fornira' in una fase successiva).
- `fk_dip_filter(data, dt_ns, dx_m, v_min, v_max, mode, taper)` — ventaglio in
  FK per velocita' apparente `f/kx`: rimuove/isola onda diretta, ground-roll,
  rumore inclinato. Testato: `reject + pass ≈ input`.
- `kirchhoff_migration_2d(data, dt_ns, dx_m, velocity_m_s, aperture_m,
  tr_offset_m)` — diffraction stack con obliquita' `cosθ` + spreading
  `1/√(v·t)`; `velocity_m_s` scalare **o** array 1D lungo n_samples
  (⇒ copre `migration_v_profile` "profile"); offset Tx-Rx bistatico.
  Testato: collassa un'iperbole sintetica all'apice (±6 campioni).
Wiring: `DEFAULT_PIPELINE` (tutti OFF), `FILTER_REGISTRY`, `DEFAULT_CHAIN_ORDER`
(19 step; `eigen_filter` dopo bg, `fk_dip` dopo bandpass, `median2d` dopo boxcar,
`kirchhoff` dopo `migration`), `_step_fingerprint`, branch `apply_pipeline`.
UI: aggiunti a `_PHASE1_SPECS` / `_PHASE1_STEP_IDS` / help → controlli generici
via `_build_phase1_step_controls`. 8 test in `test/test_gpr_processing_adv.py`
(tot 27). `migration_kirchhoff_topo` NON fatta (serve elevazione per traccia,
va con la Fase 5 navigazione/topografia).

**Fase 5 — navigazione completa + dominio griglia. ◑ CORE FATTO (2026-09).**
- `grid_filters.py` (nuovo, puro numpy, NaN-aware): `lowpass` / `hipass` /
  `laplacian` / `gaussian_smooth`; `line_background(grid, axis, length,
  clip_lo_pct, clip_hi_pct, gain)` (background di linea con soglie percentili
  per non spalmare i target); `fft2d_filter(grid, mask)` (primitiva) +
  `fft2d_directional_notch(grid, angle_deg, halfwidth_deg, mode)` (toglie
  strisce periodiche a direzione nota, es. solchi aratura); `blank(grid, mask)`;
  `mosaic_zero_mean(grids, scope, axis)` + `mosaic_histogram_match(grid, ref)`.
  15 test in `test/test_grid_filters.py`.
- `navigation.py` (nuovo, puro): `Marker` dataclass + i 4 modi
  (`artificial_markers` / `interval_markers` (+leftover) / `field_markers` /
  `gps_trace_markers`); `dead_reckon_along` / `dead_reckon_coords`
  (interpolazione lineare fra marker, estrapolazione costante ai bordi);
  `reverse_along` (linee invertite/zig-zag), `stagger_shift` (correzione
  staggering), `line_plan_coords` (ascissa→E/N per qualunque angolo, incl. ANG
  a lunghezza frazionaria con riproiezione sul segmento). 16 test in
  `test/test_navigation.py`.
- `grid_geometry.py` (nuovo, puro numpy): `rotate_translate(grid, angle_deg,
  out_origin, mode)` (ricampionamento bilineare NaN-aware; `fix_origin` porta
  l'angolo SW sorgente su `out_origin`, `translate_origin` porta l'angolo SW del
  nuovo raster); `append_grids(grids, overlap='mean'|'first'|'last')` (fonde
  gridset con stessa cella nell'inviluppo); `edge_match(a, b, axis, overlap_cells)`
  (pareggia la media nella striscia di giunzione). 9 test in
  `test/test_grid_geometry.py`.
- **Rinviato:** integrazione UI in `gpr_manual_positioning_dialog.py` (assegna i
  4 modi, mostra i marker sul radargramma, flag reverse/stagger persistenti,
  multi information-file). `migration_kirchhoff_topo` usa `line_plan_coords` +
  elevazione per traccia — resta da cablare nel viewer. `grid_geometry` va
  agganciato al percorso timeslice (Fase E del workplan).

**Fase 6 — avanzato/opzionale. ✅ FATTA (2026-09).**
Funzioni pure numpy in `gpr_processing.py`:
- `instantaneous_attributes(data, dt_ns, kind)` — dal segnale analitico
  (`_analytic_signal`): `phase` / `cos_phase` / `frequency` [MHz, da d(fase)/dt
  unwrap] / `envelope`. Wired come step pipeline `inst_attr` (registry, chain
  in coda dopo `envelope`, fingerprint, apply, factory UI con choice-param).
  Testato: la freq. istantanea di un tono a 300 MHz ≈ 300 MHz.
- `nmo_correction(data, dt_ns, offsets_m, velocity_m_s, stretch_mute)` — NMO su
  CMP/WARR gather; `offsets_m` scalare (incremento) o array; stretch mute.
  Standalone (non in pipeline: opera su gather multi-offset). Testato:
  appiattisce un'iperbole sintetica alla riga corretta.
- `cmp_semblance(gather, dt_ns, offsets_m, v_min, v_max, n_v, win_samples)` →
  `(pannello semblance (n_s, n_v), v_axis)`. Standalone. Testato: il picco di
  semblance a t0 cade sulla velocità vera (±0.6e7).
- `multifreq_blend(radargrams, weights, dt_ns_list, t0_shift_samples, target)` —
  ricampiona ogni radargramma sull'asse tempi del `target`, shift t0, somma
  pesata. Standalone (multi-input → serve variant-graph DAG o azione dedicata
  per il wiring). Testato: blend di input identici ≈ identità.
6 test nuovi in `test/test_gpr_processing_adv.py` (33 tot). Suite: 203 pass,
i 3 errori qgis preesistenti. `cmp_velocity`/`nmo` e `multifreq_blend` non hanno
ancora UI (servono un dialog CMP e un'azione blending — fase UI successiva).

---

## 9. Principi di implementazione

1. **Funzioni pure** `f(data, **params) -> data` in `gpr_processing.py`; nessun
   accesso a Qt/stato. Testabili in isolamento (`test/test_gpr_processing_adv.py`
   esiste già).
2. **`float32` in output, `float64` internamente** dove serve stabilità (FFT,
   SVD, cumsum) — come fa già `background_removal`.
3. **Logging coerente** con `_log_gpr` e diagnostica min/max/nonzero per step.
4. **Fingerprint** per ogni nuovo step in `_step_fingerprint` (necessario per la
   cache del grafo).
5. **Nessun filtro rinormalizza da solo**: la normalizzazione è display-only e
   avviene alla fine (`normalize_display`), dopo il `manual_gain`
   (cfr. `PROCESSING_FIXES.md`, Bug 1). Gain di recupero post-migrazione =
   parametro esplicito.
6. **`scope: line | survey`**: ogni filtro dichiara se sa operare
   sull'intero dataset (media/statistiche globali) oltre che per linea.
