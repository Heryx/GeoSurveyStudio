# Time-slice — analisi del disordine attuale e workplan di riscrittura

**Data:** 2026-09-08
**Da eseguire in una nuova chat.** Questo documento è autosufficiente: contiene
l'analisi dello stato attuale, l'architettura obiettivo e un piano in fasi con
i file e le firme di funzione coinvolti.

Riferimenti concettuali: **RGPR** (`GPRsurvey` + `interpSlices()`, oggetto con
asse profondità) e la separazione netta *generazione / gridding / display* dei
software di time-slicing GPR. Nessun codice di terze parti viene copiato.

Prerequisito già fatto: Fase 1 del processing (7 filtri nuovi + wiring UI) —
vedi `GPR_PROCESSING_BLUEPRINT.md`.

---

## 1. Perché oggi è "un casino"

### 1.1 Quattro generatori paralleli che si duplicano
| File | Righe | Sorgente | Duplica |
|---|---|---|---|
| `gpr_ogpr_slicer.py` | 2721 | radargrammi .ogpr / .DT | IDW, geotransform, EPSG, QML, save/load params |
| `gpr_las_slicer.py` | 530 | point cloud LAS | idem (`_apply_idw`, `_make_geotransform`, `_write_tif_*`, `save_slicer_params`) |
| `gpr_csv_slicer.py` | 741 | CSV | idem (`_slice_arrays_to_tifs`, `save_csv_slicer_params`) |
| `gpr_slice_generator.py` | 94 | PDAL | `generate_slices_pdal` |

Ogni file reimplementa binning IDW, scrittura GeoTIFF, proiezione, stile QML e
un proprio `save_*_slicer_params` / `load_*_slicer_params`.

### 1.2 Esplosione di parametri
- `compute_ogpr_slice_grids(...)` ha **~40 argomenti keyword**.
- `GprProfileViewer.get_slice_params()` (riga ~7470) costruisce un dict `extra`
  di **~30 chiavi** leggendo widget sparsi nel pannello, con un flag "Esperto"
  che **silenziosamente ignora o sovrascrive** metà dei campi.
- In parallelo `gpr_ogpr_volume_mixin._ask_ogpr_slice_params()` è un **dialog
  modale separato** con un altro set di parametri.
- `_resolve_slice_source()` restituisce una **tupla di 5 valori** per decidere
  da quale variante/step della pipeline affettare.

### 1.3 Tre funzioni di preview/compute che si sovrappongono
`compute_preview_slice()`, `compute_timeslice_preview()` (threaded via
`_PreviewSliceWorker`), `compute_ogpr_slice_grids()` / `slice_ogpr_to_tifs()`.
Logica di estrazione ampiezza + binning ripetuta con piccole varianti.

### 1.4 Un sottosistema separato per "rimettere insieme" ciò che la
generazione già sapeva
Dopo che la generazione scrive GeoTIFF in una cartella:
- `TimesliceDepthParser` (155 righe) — **indovina la profondità dal nome file**
  con regex + euristiche di coerenza + override manuali.
- `timeslice_validation_dialog` — valida CRS/georef prima dell'import.
- `project_manager_timeslice_import_mixin` — scelta CRS, group QGIS, warning georef.
- `timeslice_group_manager_dialog` (1181 righe) — rinomina, batch-rename, delete,
  assegna CRS, imposta depth range, "auto depth sequence", assegnazione gruppi, lock.
- `unassigned_timeslices_dialog` — orfani.
- Due cataloghi (slice + gruppi) con table model dedicati.

**Il nodo concettuale:** il plugin fonde due cose che vanno tenute separate:
1. **Generare** slice dai radargrammi — pipeline deterministica.
2. **Gestire** un insieme di raster-slice come stack ordinato in profondità.

La generazione conosce già indice, z_from, z_to, CRS, extent di ogni slice, poi
li **butta via** scrivendo solo GeoTIFF, e un parser fragile prova a
ricostruirli dal nome file.

---

## 2. Architettura obiettivo

### 2.1 `SliceSet` — un unico manifesto, sorgente di verità
Prodotto dalla generazione, consumato da display / 3D / export.
File `sliceset.json` accanto ai GeoTIFF:

```json
{
  "format": "rasterlinker_sliceset_v1",
  "created": "...",
  "crs_epsg": 32633,
  "source": {"kind": "ogpr", "variant_id": "bandpass", "n_profiles": 42},
  "spec": { ...SliceSpec serializzato... },
  "extent": [xmin, ymin, xmax, ymax],
  "slices": [
    {"index": 0, "z_from": 0.00, "z_to": 0.10, "z_center": 0.05,
     "grid": "slice_000.tif", "stats": {"min":..., "max":..., "p2":..., "p98":..., "fill":0.87}}
  ]
}
```

Elimina `TimesliceDepthParser` e gran parte del group manager **per i set
generati** (restano solo per raster esterni senza manifesto).

### 2.2 `slicing.py` — un unico core, 3 stadi puri e testabili
```python
def extract_amplitude(profile, z_from_m, z_to_m, mode, velocity_m_s) -> np.ndarray
    # -> (n_traces,) scalare per traccia
    # mode: "abs_mean" | "rms" | "env_peak" | "signed_mean" | "max_abs"

def build_slice_points(survey, spec) -> list[SlicePoints]
    # per ogni finestra di profondità: (x, y, value) da tutte le tracce di tutti i profili
    # gestisce overlap, thickness, topografia

def grid_slice(points, grid_spec, interp) -> np.ndarray
    # IDW (kNN + ball) o kriging; + low-pass opzionale + blanking da distanza
```
I 4 slicer diventano **adapter di input** sottili che producono `survey`
(profili + geometria) e delegano a stage 2/3. Codice IDW/GeoTIFF/QML/params
condiviso in `slicing.py` + un `raster_io.py` unico.

### 2.3 `SliceSpec` — un unico dataclass di parametri
~12 campi con default sensati, raggruppati:

```python
@dataclass
class SliceSpec:
    # profondità
    z_min_m: float = 0.0
    z_max_m: float | None = None      # None -> profondità max dei profili
    thickness_m: float = 0.10
    overlap: float = 0.5              # 0..0.9
    # estrazione
    amplitude_mode: str = "abs_mean"
    source_variant: str = "raw"       # id variante pipeline / "raw"
    velocity_m_s: float = 1e8
    # grid
    resolution_m: float = 0.10
    interp: str = "idw"              # "idw" | "kriging"
    search_radius_m: float | None = None   # None -> auto da spaziatura interlinea
    blanking_radius_m: float = 0.0
    smooth_lowpass: int = 0          # 0 | 3 | 5
    # crs
    crs_epsg: int | None = None
```

Sostituisce `get_slice_params()` + `_ask_ogpr_slice_params()` + flag Esperto con
**un solo form** legato al dataclass, persistito in `sliceset.json`.

### 2.4 Un unico percorso di preview
`grid_slice()` a risoluzione grossa **è** la preview. Si eliminano
`compute_preview_slice()` e `compute_timeslice_preview()`; `_PreviewSliceWorker`
chiama lo stesso core con `resolution` grossolana e una sola finestra.

### 2.5 Import raster esterni = solo fallback
`TimesliceDepthParser` + `timeslice_group_manager_dialog` restano **solo** per
GeoTIFF prodotti fuori dal plugin (senza `sliceset.json`). Un set generato salta
validazione-CRS, parsing-nome, assegnazione-gruppo manuale.

### 2.6 Display / 3D leggono il `SliceSet`
`_slice_catalog`, i table model dei gruppi e `build_3d_volume` consumano il
manifesto: asse profondità e ordinamento sono noti, niente ricostruzione.

---

## 3. Workplan in fasi

Ogni fase è un commit autonomo, con test headless dove possibile
(`test/test_*_slicing*.py`). Non toccare la UI finché il core non è verde.

### Fase A — estrarre il core condiviso (nessun cambio di comportamento)
1. Creare `raster_io.py`: `make_geotransform`, `set_projection`, `write_tif_singleband`,
   `write_tif_rgb`, `write_qml_singleband`, `write_qml_multiband` — spostati e
   deduplicati da `gpr_ogpr_slicer.py` / `gpr_las_slicer.py` / `gpr_csv_slicer.py`.
2. Creare `slicing.py` con `extract_amplitude`, `build_slice_points`, `grid_slice`
   estraendo la logica **già esistente** in `gpr_ogpr_slicer.py`
   (`_envelope`, `_depth_to_sample_range`, `_bin_with_idw*`, `_fill_nodata_grid`,
   `_smooth_grid_gaussian`, `_blanking_mask_from_distance`, `_depth_adaptive_radius`).
3. `gpr_ogpr_slicer.py`, `gpr_las_slicer.py`, `gpr_csv_slicer.py` reimportano da
   `raster_io` / `slicing` invece di duplicare. Le firme pubbliche
   (`slice_ogpr_to_tifs`, `slice_las_to_tifs`, `slice_csv_to_tifs`) restano.
4. Test: parità numerica prima/dopo su un dataset sintetico piccolo.

### Fase B — `SliceSpec` dataclass + `sliceset.json`
1. `slice_spec.py`: dataclass `SliceSpec` + `to_json`/`from_json` + `resolve()`
   (riempie `z_max_m`, `search_radius_m`, `crs_epsg` dai profili).
2. `slice_manifest.py`: `write_sliceset(dir, spec, slices, source)` /
   `read_sliceset(dir)` — formato `rasterlinker_sliceset_v1`.
3. `compute_ogpr_slice_grids` accetta **anche** `spec: SliceSpec | None`; se
   presente, ignora i ~40 kwarg legacy. Mantieni i kwarg per retrocompat.
4. `slice_ogpr_to_tifs` scrive `sliceset.json` accanto ai GeoTIFF.
5. Test: round-trip spec, manifest coerente con i file scritti.

### Fase C — collassare le 3 funzioni di preview in una ✅ CORE FATTO (2026-09)
1. ✅ `compute_slice(profiles, spec, *, single_window=None, resolution_override=None,
   channel, combine_method, pipeline_params, use_processing, normalize_channels,
   **extra) -> (grids, meta)` in `gpr_ogpr_slicer.py`. Risolve lo spec, con
   `single_window=(z0,z1)` forza **una** slice (unico centro `zc`, `z_max` appena
   sopra `zc` così `np.arange` dei livelli ne produce uno solo), `resolution_override`
   forza la risoluzione. `extra` = passthrough a `compute_ogpr_slice_grids`.
2. ✅ `compute_slice_preview(profiles, spec, z_from, z_to, *, resolution=None,
   **extra) -> (grid float32, extent)` — rimpiazzo spec-native di
   `compute_timeslice_preview` (extent = bordi esterni pixel, come
   `raster_io.make_geotransform`).
3. ⏳ **RINVIATO** `_PreviewSliceWorker.run()` → `compute_slice_preview(...)`:
   vive in `gpr_profile_viewer.py`, si fa insieme alla Fase D (serve la
   `SliceSpec` dal nuovo pannello). Le vecchie `compute_preview_slice` /
   `compute_timeslice_preview` restano invariate finché la UI non è migrata.
4. ✅ `test/test_compute_slice.py` (6 test): type-guard, geometria multi-slice
   (i "z levels" sono CENTRI da z_min a z_max, passo = thickness·(1−overlap)),
   `single_window` → 1 sola slice con z_from/z_to = finestra, `resolution_override`,
   **parità** `compute_slice_preview` == `compute_slice(single_window=…)` stesso
   array a pari risoluzione, finestra NaN → `(None, None)`.

### Fase D — UI: un solo form legato a `SliceSpec`
1. Nuovo `slice_spec_panel.py`: `SliceSpecPanel(QWidget)` con ~12 controlli
   mappati 1:1 al dataclass, `get_spec()` / `set_spec()`.
2. Nel `GprProfileViewer`: sostituire il blocco Timeslice del pannello e
   `get_slice_params()` con `self._slice_spec_panel.get_spec()`.
3. Rimuovere il flag "Esperto" e `_ask_ogpr_slice_params` (il form è sempre
   visibile e completo). `_resolve_slice_source` → `spec.source_variant`.
4. Persistenza: salvare/caricare `SliceSpec` in `GPR_Processing/project.json`.

### Fase E — display / 3D leggono il manifesto
1. `_slice_catalog` popolato da `read_sliceset()` quando esiste; fallback al
   comportamento attuale (scan cartella + `TimesliceDepthParser`) solo senza
   manifesto.
2. `timeslice_group_manager_dialog`: se il set ha `sliceset.json`, disabilita
   "auto depth sequence" / "set depth range" (già noti) e mostra solo
   rinomina/gruppi. Marcare il resto come *solo raster esterni*.
3. `build_3d_volume` prende `SliceSet` (asse z dal manifesto) invece di
   dedurre dz dai nomi.
4. `project_manager_timeslice_import_mixin`: import diretto senza dialog di
   validazione CRS quando `sliceset.json` porta un `crs_epsg` valido.

### Fase F — unificare gli adapter LAS/CSV/PDAL
1. `gpr_las_slicer` / `gpr_csv_slicer` / `gpr_slice_generator`: ridurli a
   `read_*_as_survey(path/params) -> Survey` + chiamata a `compute_slice`.
2. Un solo `save_slice_params` / `load_slice_params` (serializza `SliceSpec`).
3. Eliminare `_apply_idw` / `_make_geotransform` / `_write_tif_*` locali.

---

## 3bis. Esecuzione residua D / E / F — precondizioni e punti di innesto

Le fasi D, E, F sono UI/integrazione pesante su file che l'utente sta
**attivamente modificando** (rework IDW in `gpr_ogpr_slicer.py`, più
`gpr_las_slicer.py` / `gpr_csv_slicer.py` / `gpr_profile_viewer.py` /
`project_manager_timeslice_import_mixin.py` / `timeslice_validation_dialog.py`
tutti `M`). Non vanno fatte alla cieca: servono **(1)** il working tree ripulito
(rework IDW committato o splittato) e **(2)** verifica interattiva in QGIS.
Punti di innesto già mappati:

**Fase D — pannello `SliceSpec`**
- Nuovo `slice_spec_panel.py`: `SliceSpecPanel(QWidget)`, 13 controlli 1:1 con
  `slice_spec.SliceSpec`, `get_spec() -> SliceSpec` / `set_spec(SliceSpec)`.
  Mappare `smooth_lowpass` (0/3/5) e `amplitude_mode`
  (`abs_mean`/`rms`/`env_peak`/`signed_mean`/`max_abs`) su combo.
- In `gpr_profile_viewer.py`: il blocco "Timeslice" del pannello
  (`_build_processing_panel`, da ~riga 1985) e **tutto** `get_slice_params()`
  (~riga 7470) → `self._slice_spec_panel.get_spec().resolve(self._profiles)`.
  `_resolve_slice_source()` (tupla 5-valori) → un solo campo
  `spec.source_variant`. Togliere il flag "Esperto" (`_btn_slice_expert`) e
  `gpr_ogpr_volume_mixin._ask_ogpr_slice_params`.
- `_PreviewSliceWorker.run()` (~riga 398) → `compute_slice_preview(profiles,
  spec, z0, z1, resolution=preview_res)` (già pronto, Fase C).
- Le chiamate finali: `slice_ogpr_to_tifs(..., spec=spec, source_variant=…)` e
  `compute_ogpr_slice_grids(..., spec=spec)` (entrambe già accettano `spec`).
- Persistenza: `SliceSpec.to_dict()` in `GPR_Processing/project.json`
  (`_save_gpr_processing_project_meta` / `_load_gpr_processing_project_meta`).

**Fase E — filtri/geometria griglia + display dal manifesto**
- ✅ (2026-09, passo UI 5) `raster_io.read_tif_singleband(path)` — controparte di
  `write_tif_singleband` (NODATA→NaN, righe rifless.). Azione viewer
  `_filter_timeslices_grid`: filtra un set di GeoTIFF con `grid_filters`
  (lowpass 3/5/7, hipass, laplacian, background di linea X/Y con clip percentile,
  notch FFT direzionale) → nuovo set `<cartella>_<suffisso>/`, copia il
  `sliceset.json`, ripunta `_le_slice_outdir` + rinfresca il catalogo. Menu
  Processing → "Filtra time-slice (dominio griglia)...". Nuovo tipo controllo
  `bool` nella factory già aggiunto al passo 4. `grid_geometry`
  (rotate/translate/append/edge_match) NON ancora esposto in UI.
- ⏳ `slice_manifest.read_sliceset(dir)` restituisce già `slices` ordinate con
  `index/z_from/z_to/z_center/grid/stats` + `spec` (SliceSpec) + `crs_epsg`.
  Aggiungere un helper `slice_catalog_from_sliceset(dir)` che converte in righe
  `_slice_catalog` (path assoluto = `dir/grid`, campi depth dal manifesto).
- `_slice_catalog` (viewer + `timeslice_group_manager_dialog`): se
  `has_sliceset(dir)` usare quell'helper; altrimenti fallback scan cartella +
  `TimesliceDepthParser`.
- `timeslice_group_manager_dialog`: con `sliceset.json` presente disabilitare
  "auto depth sequence" / "set depth range" (`_auto_depth_sequence_selected`,
  `_set_depth_range_selected`) — profondità già autorevole.
- `build_3d_volume` / `gpr_volume_3d`: accettare l'asse z dal manifesto invece
  di dedurre `dz` dai nomi.
- `project_manager_timeslice_import_mixin._validate_timeslice_records_before_import`:
  saltare il dialog CRS quando `sliceset.json` porta `crs_epsg` valido.

**Fase F — adapter LAS / CSV / PDAL**
- `gpr_las_slicer` / `gpr_csv_slicer` / `gpr_slice_generator`: ridurre a
  `read_*_as_profiles(path/params) -> list[profile-like]` (stessa interfaccia
  minima di `_FakeProfile`: `.channel(i).data/easting/northing/altitude`,
  `.depth_max_m`, `.epsg`) + `compute_slice(profiles, spec, …)`.
- Un solo `save_slice_params` / `load_slice_params` = `SliceSpec.to_json` /
  `from_json` (togliere `save_las_slicer_params` / `save_csv_slicer_params` /
  `save_slicer_params`).
- Eliminare `_apply_idw` / `_make_geotransform` / `_write_tif_*` locali → usare
  `slicing` + `raster_io` (già estratti in Fase A).

## 3ter. Fase G — motore "bin" (2026-09-21): puntini, colormap, preview lenta

Segnalazione dell'utente (rilievo SUASA, array IDS a 19 canali): puntini
regolari sulle slice, colormap non cambiabile, preview molto lenta. Diagnosi con
i dati reali (25 swath `.npz`, variante `Array1_presetIDS`):

- **Puntini** = collana di celle isolate lungo ogni linea di rilievo. Causa:
  `_process_profiles` mediava coordinate *e* ampiezze dei 19 canali su una sola
  linea centrale per swath (4 cm di passo laterale, ~0.75 m di swath buttati) e
  poi IDW (peso `1/d^p`, singolare sui punti) interpolava tutto il resto.
- **Profondita' sbagliata**: `_depth_to_sample_range` usava `righe/depth_max`, ma
  dopo time-zero + `_align_2d_arrays_for_stack` le matrici hanno 865-1024 righe
  (non 985): la slice "4.08 m" mostrava cio' che il radargramma ha a ~3.6 m.
  `SliceSpec.velocity_m_s` non era usata da nessuno.
- **Bande lungo le tracce** (mediane per canale da 0.19 a 0.65): nascoste dalla
  media dei canali, emergono appena si usa la risoluzione vera dell'array.
- **Colormap**: `_cb_slice_cmap` stava nell'"Avanzato" legacy, nascosto dalla
  Fase D. Stesso problema per il clip percentile. Lo spessore della preview
  leggeva uno spinbox nascosto non collegato al pannello.
- **Preview lenta**: 13.6 s (IDW a ball query con ciclo Python per cella +
  `_process_profiles` completo a ogni cursore) e la prima volta la GUI restava
  bloccata leggendo ~1 GB di `.npz` nel thread principale.

Fatto:

- `slicing.py`: `window_amplitude` (5 modi veri: `signed_mean` ora ha il segno,
  `abs_mean` e' davvero la media di |a|, `env_peak` = picco dell'inviluppo su
  finestra + contesto), `window_sample_range`/`depth_per_sample_m` (profondita' =
  `riga * dt * v / 2`, come l'asse del radargramma), `grid_binned` (splat
  bilineare + convoluzione gaussiana normalizzata a due scale + chiusura dei
  buchi con EDT: niente KD-tree, niente cicli per cella).
- `slice_engine.py` (nuovo): ogni canale alla sua posizione, equalizzazione per
  canale/profilo (`SliceSpec.balance`), finestre `[z_min + i*passo, +spessore]`
  interamente dentro `[z_min, z_max]`, `(grids, meta)` compatibile con
  `compute_ogpr_slice_grids` (tif, manifest, 3D invariati). Georef esatta
  (`y_max = y_min + (n_y-1)*res`: il legacy poteva sfasare di una cella).
- `SliceSpec`: `interp` default `"bin"` (`"idw"` = percorso legacy, invariato e
  testato), `balance`, `velocity_m_s=None` = velocita' del profilo, low-pass
  default 3x3. Uno spec salvato prima di questa fase (senza `balance`) con
  `interp="idw"` passa a `"bin"` al caricamento.
- Preview: ~0.2 s a caldo (era 13.6 s); la variante si legge nel thread di
  lavoro, una sola volta (`_preview_loading_key`), con cache.
- UI: colormap + anteprima + contrasto (clip %) visibili nel tab Slicing e
  salvati nel progetto (`ui_state.slice_style`); spessore del pannello e'
  l'unica fonte (`_sync_thickness_from_panel`); combo Interpolazione/
  Bilanciamento/Velocita' auto nel pannello.
- Output: il `.qml` accanto ai GeoTIFF e' ora `singlebandpseudocolor` con la
  colormap e il clip scelti (11 stop; fallback al grigio se la colormap non e'
  risolvibile). Vale anche per le slice filtrate/rototraslate.
- Test: `test/test_slice_engine.py` (37, headless) e `test/test_viewer_timeslice.py`
  (9, Qt offscreen sotto QGIS). Verifica su dati reali dentro QGIS 3.36: 60
  slice 0-6 m in 11 s, il 100% delle tracce cade su pixel validi, QML caricato.

Non fatto / da valutare:

- Fase E (leggere il manifest per catalogo/gruppi/3D) e Fase F (adapter LAS/CSV)
  restano come sopra.
- Stretch comune a tutto il set (oggi ogni slice ha il proprio 2-98 %).
- Denoise fisico oltre la gaussiana (es. filtro lungo-traccia, kriging).
- I nomi file `slice_NNNN_zX_XXXX.tif` (centro della finestra) non sono cambiati.

## 4. Ordine consigliato e rischio
- A, B, C (core): ✅ fatte. Rischio basso, testate headless.
- D: medio-alto (pannello grande `gpr_profile_viewer.py`), isolabile in
  `slice_spec_panel.py`. Serve QGIS per verificare.
- E, F: dipendono da B/C; E rende utile il manifesto, F chiude la duplicazione.
  F tocca file che l'utente sta modificando → dopo il commit del rework IDW.

## 5. Definizione di "fatto"
- Un solo modulo core (`slicing.py` + `raster_io.py`) usato da tutti i sorgenti.
- Un solo dataclass di parametri, un solo form, una sola funzione di compute.
- Ogni set generato ha `sliceset.json`; profondità e ordinamento mai più
  dedotti dai nomi file.
- `TimesliceDepthParser` e il grosso di `timeslice_group_manager_dialog` usati
  solo per raster esterni.
- Test headless per estrazione ampiezza, binning, gridding, manifesto.
