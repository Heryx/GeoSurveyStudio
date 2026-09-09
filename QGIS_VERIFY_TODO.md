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

---

## Passo 8 — Quote per traccia nel Posizionamento Manuale (commit da pushare)

Profile Viewer → menu → **Posizionamento Manuale Griglia GPR** (profili senza
GPS). Nel gruppo **Navigazione** c'è ora "Quota per traccia".

1. **Piano inclinato:** selezionare il modo, impostare *Quota inizio linea* e
   *Quota fine linea* (es. 100 → 105). Applicare.
   **Atteso:** `ch.altitude` di ogni profilo varia linearmente lungo la linea;
   su una linea in zig-zag (invertita) la prima traccia acquisita prende la
   quota dell'estremo "fine" della linea di griglia.
2. **Da file:** preparare un `.csv`/`.txt` a due colonne (`posizione, quota`);
   scegliere se la colonna 1 è *metri lungo la linea* o *indice di traccia*.
   Applicare. **Atteso:** quote interpolate (`np.interp`) sulle tracce; file
   illeggibile → messaggio d'errore, nessuna modifica.
3. **Nessuna quota** (default): `ch.altitude` **non** viene toccato (si
   preserva un'eventuale quota da `.gec`/GPS).
4. Riaprire il dialog: modo quota, quote inizio/fine, path file e
   interpretazione colonna devono risultare **ripristinati**.
5. Verifica d'uso: con le quote assegnate, la **Migrazione Kirchhoff
   topografica** (passo 4) deve usarle senza bisogno del `.gec`.

---

## Passo 9 — Modi marker + overlay nel Posizionamento Manuale (commit da pushare)

Stesso dialog, gruppo **Navigazione** → "Modo posizionamento", e nuovo pannello
a destra **"Radargramma + marker (profilo selezionato)"**.

1. Selezionare una riga nella tabella profili: il pannello mostra il
   radargramma di quel profilo (canale 0, scala di grigi).
2. **Uniforme (griglia)** (default): nessun marker disegnato; il calcolo è
   `linspace` come prima → risultato identico all'attuale.
3. **Interval marker (passo scan fisso):** impostare "Scan per marker"
   (0 = automatico, ~1/metro). **Atteso:** linee verticali rosse sui marker;
   titolo/etichetta indicano quanti marker e quanti scan di **coda** oltre
   l'ultimo marker vengono scartati ("sbordo"). Le coordinate delle tracce
   vengono da dead‑reckoning fra i marker.
4. **Artificial marker (N equidistanti):** impostare "N marker". **Atteso:**
   N linee equidistanti; senza calibrazione ruota il risultato numerico è
   ~uguale all'uniforme (è la conferma attesa, non un bug).
5. Cambiare selezione / modo / parametro deve aggiornare subito il disegno.
6. Riaprire il dialog: modo e parametro devono risultare **ripristinati**.
7. Con `matplotlib` assente il pannello mostra un testo di fallback, il resto
   del dialog funziona.
8. **Baseline di ritorno (zig-zag):** con "Usa la lunghezza reale" attivo e
   profili di ritorno più corti dello span, attivare "Linee invertite: parti
   dalla baseline di ritorno". **Atteso** nell'anteprima schema: le linee
   pari (ritorno) partono allineate al lato **destro** (x1) / **alto** (y1)
   ed estendono all'indietro, invece di essere rientrate dal lato di
   partenza. Con la checkbox off → comportamento attuale. Lo stagger si
   somma comunque a questa scelta. Stato ripristinato alla riapertura.

---

## Fix — persistenza processing + Passo 10 (Varianti ad albero)

### Persistenza stato di processing
1. Processare un profilo (gain, dewow, bandpass, bg, ecc.), **chiudere** il
   Profile Viewer, riaprirlo / riaprire il progetto GPR. **Atteso:** i
   controlli di processing e l'anteprima tornano com'erano, **senza**
   riprocessare a mano; il file `GPR_Processing/project.json` contiene una
   sezione `processing_state`.
2. Cambiare qualche parametro e richiudere → alla riapertura si ritrova
   l'ultima configurazione.

### Passo 10 — pannello Varianti ad albero
1. Salvare una variante (menu → "Salva variante processata", es. `bp`).
   Nel pannello **Varianti** compare come nodo radice; `processed/` contiene
   `variant_graph.json`.
2. Sul nodo `bp` → tasto destro → **"Nuova variante da qui"**: i controlli di
   processing si popolano con la pipeline di `bp`; barra di stato conferma che
   la prossima variante sarà registrata come figlia.
3. Aggiungere un filtro e salvare come `bp_bg`. **Atteso:** nell'albero
   `bp_bg` appare **annidato sotto** `bp`; colonna "Ultimo step" coerente.
4. Doppio click su un nodo → genera time-slice (come prima). "Apri cartella"
   e "Elimina variante…" dal menu contestuale funzionano; eliminando un nodo
   con figli, i figli si ri-agganciano al nonno.
5. Rinominare a mano una cartella variante o cancellare `variant_graph.json`:
   il pannello non deve andare in errore (le cartelle senza voce di
   provenienza diventano radici).
