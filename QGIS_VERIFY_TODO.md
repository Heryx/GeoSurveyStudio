# Da verificare in QGIS — passi UI non ancora testati sul campo

**Aggiornato:** 2026-09-09
**Branch:** `feature/interpolation-improvements-followup`

Elenco delle modifiche già committate ma **non ancora provate dentro QGIS**.
Man mano che una verifica passa, spostare la voce in `UI_INTEGRATION_TODO.md`
nella lista "Passi UI fatti (verificati in QGIS dall'utente)".

---

## Passo 6 — `grid_geometry` in UI (commit `b064ed7`)

Menu **Processing** del Profile Viewer, due nuove voci.

### 6a — "Unisci set di time-slice…"
Serve un dataset con **due griglie adiacenti dello stesso sito** (stessa cell
size, stessi indici z). *(In attesa che l'utente recuperi i dati.)*

1. Generare/avere 2 cartelle di GeoTIFF di time-slice.
2. Processing → **Unisci set di time-slice…** → selezionare la prima cartella,
   poi la seconda (Annulla per terminare / rispondere "No" alla richiesta di
   aggiungerne altre).
3. Nel dialog: lasciare **Edit-match** attivo, asse giunzione coerente con la
   disposizione (X = affiancate, Y = impilate), `mean` sulle sovrapposizioni.
4. **Atteso:** cartella `<prima>_merged/` con un set unico; caricato nella
   vista Slicing **senza salto di livello alla giunzione**; allineamento
   corretto in QGIS; `sliceset.json` unito presente.

### 6b — "Ruota/trasla set di time-slice…"
1. Avere una cartella di time-slice (anche una sola griglia locale).
2. Processing → **Ruota/trasla set di time-slice…** → impostare `angolo`,
   `origine E/N`, `modo` (`fix_origin` per portare l'angolo SW sull'origine).
3. **Atteso:** cartella `<cartella>_rt/` con le slice ruotate/traslate,
   ricampionamento bilineare pulito, georeferenziazione corretta in QGIS,
   `sliceset.json` con `extent` ricalcolato.

---

## Passo 7 — SliceSet dal manifesto (commit `f327eba`)

1. Generare un set di time-slice (deve scrivere `sliceset.json` accanto ai
   `.tif`).
2. **Rinominare a caso** alcuni file `.tif` del set (senza toccare il
   `sliceset.json`).
3. Aprire/aggiornare la vista Slicing su quella cartella.
   **Atteso:** navigatore profondità e titoli slice mostrano comunque le
   profondità **corrette** (lette dal manifesto, non dai nomi).
4. Importare il set nel progetto: **atteso** nessun prompt CRS se il manifesto
   ha un `crs_epsg` valido (anche se i GeoTIFF non hanno CRS embedded).
5. Aprire il **Time-slice Group Manager**, selezionare le slice di quel set:
   **atteso** i pulsanti *Set Depth Range* e *Auto Depth Sequence* disabilitati,
   con nota "Profondità da sliceset.json…" sotto la tabella. Premendoli
   (se abilitati per errore) deve comparire il messaggio di blocco.
6. Controprova: un set **senza** `sliceset.json` deve comportarsi come prima
   (parsing nomi, pulsanti profondità attivi).
