# UI integration — cosa resta da fare

**Aggiornato:** 2026-09-09
**Branch:** `feature/interpolation-improvements-followup` (tutto committato e pushato
su `geosurveystudio`)
**Test:** `python -m unittest discover -s test` → 249, i 3 errori
`test_qgis_environment / test_resources / test_translations` sono **preesistenti**
(servono un runtime QGIS), tutto il resto verde.

Contesto completo: `GPR_PROCESSING_BLUEPRINT.md` (motore processing, Fasi 1‑6) e
`TIMESLICE_REDESIGN_WORKPLAN.md` (time‑slice, Fasi A‑F). Direzione e storico in
memoria: `project_processing_rewrite.md`, `project_timeslice_redesign.md`.

---

## Fatto — core headless (Fasi 1‑6 + §5.4) e passi UI 1‑9

**Core (puro numpy, testato headless):**
- Fase 1: `dcshift`, `boxcar_smooth`, `notch_filter`, `spectral_whitening`,
  `amplitude_equalization`, `radar_threshold`, `power_gain`
- Fase 2: `bp_taper_frac`; fit iperbole bistatico (offset Tx‑Rx) + readout
- Fase 3: `variant_graph.py` (`VariantGraph`, `resolve`, `materialize_arrays`)
- Fase 4: `eigen_filter`, `median_filter2d`, `fk_filter`, `fk_dip_filter`,
  `kirchhoff_migration_2d`
- Fase 5: `grid_filters.py`, `navigation.py`; §5.4 `grid_geometry.py`
- Fase 6: `instantaneous_attributes` (step `inst_attr`), `nmo_correction`,
  `cmp_semblance`, `multifreq_blend`
- Extra: `kirchhoff_migration_topo` (+ tipo controllo `bool` nella factory UI)

Tutti i filtri per‑traccia sono nel `DEFAULT_CHAIN_ORDER` e hanno controlli
generati da `_PHASE1_SPECS` / `_build_phase1_step_controls` in
`gpr_profile_viewer.py`.

**Passi UI fatti (verificati in QGIS dall'utente):**
1. Tab **"Varianti"** nel viewer — `bf32938`
2. Posizionamento manuale: **lunghezza reale** + **stagger** — `08431a1`, `da329d2`
3. **Combina multi‑frequenza → variante** (canali di un file *o* profili
   separati LF/HF) — `03fd94c`, `c71c7e5`
4. **Migrazione Kirchhoff topografica** (usa le quote per traccia dal `.gec`) —
   `37b2a43`
5. **Filtra time‑slice (dominio griglia)** — `raster_io.read_tif_singleband` +
   azione menu che applica `grid_filters` a un set di GeoTIFF — `7a1f163`

**Fatti, DA verificare in QGIS** (checklist: `QGIS_VERIFY_TODO.md`):

6. **Unisci / Ruota‑trasla set di time‑slice** — `grid_geometry` in UI
   (`append_grids` + `edge_match` per l'unione multi‑griglia, `rotate_translate`
   per la georeferenziazione); manifesto unito/aggiornato — `b064ed7`
7. **SliceSet dal manifesto** — `slice_manifest.slice_catalog_from_sliceset`;
   il viewer legge profondità/ordine/nomi da `sliceset.json` quando presente
   (fallback al parsing nomi); group‑manager blocca l'editing profondità sui
   set con manifesto; l'import salta il prompt CRS se il manifesto porta un
   `crs_epsg` valido — vedi sotto per il punto 3D rimandato
8. **Quote per traccia nel Posizionamento Manuale** — `compute_positions`
   ritorna anche `elev`; gruppo "Navigazione" con modo quota *nessuna* /
   *piano inclinato* (quota inizio→fine, segue lo zig-zag) / *da file*
   (`x,z` o `traccia,z`); `ch.altitude` assegnato solo se il modo ≠ nessuna;
   stato persistito in `navigation` e ripristinato alla riapertura
9. **Modi marker + overlay nel Posizionamento Manuale** — `navigation.build_markers`
   (dispatcher uniform/interval/artificial); `compute_positions(..., markers=)`
   usa `dead_reckon_along` invece di `linspace`; combo "Modo posizionamento" +
   parametro passo/N nel gruppo Navigazione; nuovo modulo `radargram_render.py`
   (`draw_radargram` condiviso) e mini-canvas radargramma+marker nel dialog che
   mostra il profilo selezionato e la coda "sbordata"; stato in `navigation`
   (`pos_mode`/`pos_param`) ripristinato alla riapertura.
   Inoltre `compute_positions(..., reverse_anchor="far")` + checkbox "Linee
   invertite: parti dalla baseline di ritorno": le linee di ritorno più corte
   dello span vengono allineate al lato opposto (x1/y1) invece che rientrate
   dal lato di partenza (zig-zag di un'area rettangolare).

**Fix** — persistenza dello stato di processing: la ricetta completa della
pipeline (parametri per-step + ordine/enable catena + gain + iperbole) viene
salvata in `GPR_Processing/project.json` (`processing_state`) e ri-applicata
all'apertura del progetto, con un solo riprocessamento. Non serve più
riprocessare tutto da zero a ogni riapertura del Profile Viewer.

3b (dialog CMP semblance) **saltato** su richiesta utente (acquisizioni
WARR/CMP di nicchia per l'archeologia). Le funzioni `cmp_semblance` /
`nmo_correction` restano disponibili come API.

---

## Da fare — in ordine consigliato

### Passo 7bis (rimandato) — asse z 3D dal manifesto
`build_3d_volume` / `volume_axes` (`gpr_volume_3d.py`) **già** accettano
`z_from`/`z_to`/`z_lev` per griglia e `z_levels` nel meta; il gap teorico
("dedurre `dz` dai nomi") non ha oggi un chiamante concreto: il volume 3D è
sempre ricalcolato dai profili via `compute_ogpr_slice_grids`, non da una
cartella di GeoTIFF esistente. Da fare **solo** se in futuro si aggiunge un
"costruisci volume da set di time‑slice su disco": allora leggere l'asse z da
`slice_manifest.read_sliceset(dir)["slices"]` e passarlo come `z_levels`.

### Passo 10 — Pannello Varianti come grafo `VariantGraph`  ·  valore medio  ·  rischio medio
Oggi la scheda "Varianti" è una **lista flat** di cartelle `processed/<nome>/`.
Portarla al modello a grafo di `variant_graph.py`:
- Persistere `variant_graph.json` in `processed/` quando si salva una variante
  (`_save_processed_variant`): nodo con `parent` = variante di partenza, `step` =
  ultimo filtro attivo, `params`.
- Scheda "Varianti" → `QTreeWidget` dell'albero; menu contestuale "nuova variante
  da qui" (carica quella come input della pipeline), "materializza".
- `_slice_from_variant_folder` invariato (lavora sui `.npz`).

### Passo 11 — Fase F: unificare adapter LAS / CSV / PDAL  ·  strutturale  ·  headless‑friendly
Dettagli in `TIMESLICE_REDESIGN_WORKPLAN.md §3bis` (Fase F).
- `gpr_las_slicer` / `gpr_csv_slicer` / `gpr_slice_generator` → ridurre a
  `read_*_as_profiles(...) -> list[profile-like]` + `compute_slice(profiles,
  spec, …)`.
- Un solo `save_slice_params` / `load_slice_params` = `SliceSpec.to_json` /
  `from_json`.
- Eliminare `_apply_idw` / `_make_geotransform` / `_write_tif_*` locali → usare
  `slicing.py` + `raster_io.py`.
- Test di parità prima/dopo (come `test_slicing_core.py`).

---

## Come procedere in una nuova sessione
1. Leggere questo file + `GPR_PROCESSING_BLUEPRINT.md` §8 + `TIMESLICE_REDESIGN_WORKPLAN.md` §3bis.
2. Confermare tree pulito su `feature/interpolation-improvements-followup`,
   `python -m unittest discover -s test` = 223 (3 errori qgis preesistenti).
3. Un passo alla volta: implementare → `py_compile` + suite → commit + push →
   l'utente verifica in QGIS → passo successivo.
4. Ogni commit chiude con `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
