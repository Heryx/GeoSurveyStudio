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

### Fix — "Variante sorgente" selezionabile nel pannello Slicing
1. Con almeno una variante salvata, aprire la tab **Slicing**: il campo
   "Variante sorgente" e' una **tendina** con `raw` + i nomi delle varianti.
2. Selezionare una variante e premere **"Salva gruppo"**. **Atteso:** le
   time-slice vengono generate dai `.npz` **gia' filtrati** di quella variante
   (nessun ri-processing), output in `GPR_Processing/timeslices/<variante>/`.
3. Con "Variante sorgente" = `raw` → comportamento invariato (affetta i profili
   correnti).
4. "Apri Viewer 3D" con una variante selezionata usa gli stessi dati filtrati.
5. Salvare una nuova variante → compare subito nella tendina.

---

## Pannello "Radargrammi processati" nel canvas (2026-09-21, non committato)

Nuovo pannello agganciabile (`gpr_radargram_dock.py` + `radargram_store.py`), indipendente
dal Profile Viewer. Testato offscreen con un `QgsMapCanvas` vero e sul dataset SUASA;
**non** provato nella finestra reale di QGIS. Ricaricare il plugin prima.

1. Toolbar/menu GeoSurvey Studio → **Radargrammi processati**: il pannello compare in
   basso; trascinandolo si aggancia a un lato o si stacca (finestra mobile). Da stretto
   i controlli vanno a capo.
2. Con il Profile Viewer **chiuso**: variante (default "Lavoro"), profilo e canale si
   cambiano da tendine, ◀ ▶, PgSu/PgGiu (profilo) e frecce sinistra/destra (canale).
3. **Profondità**: clic sul radargramma o campo "Profondità" → riga verde; il cursore
   mostra profondità, tempo (ns), traccia, ampiezza, E/N. Rotella = zoom, tasto destro =
   trascina. **[cambiato, vedi sezione in fondo]** doppio clic non fa più "vista
   intera" (ora c'è il pulsante dedicato ⤢): attiva la time-slice alla profondità
   cliccata.
4. Selezionare una time-slice nel pannello Livelli (con "Segui time-slice"): **atteso**
   fascia verde = finestra di profondità della slice e passaggio alla variante da cui
   è stata generata. Le slice senza `sliceset.json` accanto (gruppi vecchi) non fanno nulla.
5. **Sulla mappa**: linea arancione = traccia del profilo/canale; punto giallo sotto il
   cursore. Controllare che coincidano con la time-slice (stesso CRS del progetto).
6. **Punta sulla mappa**: un clic sul canvas apre profilo/canale/traccia più vicini
   (tolleranza ~10 px, minimo 2 m); scegliendo un altro strumento QGIS il pulsante si spegne.
7. Chiudere il pannello con la X: linea e punto spariscono dalla mappa; riaprirlo li rimette.
8. Generare un gruppo di time-slice ("Salva gruppo"): **atteso** il pannello si apre (senza
   rubare il focus) sulla variante sorgente.
9. Riga di stato: "coordinate: salvate nel .npz" se il viewer è chiuso; assente se è aperto
   (geometria dai profili in memoria). Dopo un nuovo posizionamento premere ⟳.

---

## Disegno anomalie/interfacce sul radargramma + puntatore velocità (2026-09-22, non committato)

Nel pannello "Radargrammi processati": nuovi pulsanti **Punto / Linea / Poligono**
(disegno) e **Misura velocità** (fit iperbole live), riga di controlli sopra il
radargramma. Testato offscreen (`test_radargram_dock.py`, `test_radargram_capture.py`
`test_hyperbola_math.py`); **non** provato nella finestra reale di QGIS. Ricaricare il
plugin prima.

1. **Linea**: attivare "Linea", cliccare 3-4 punti seguendo un'interfaccia/anomalia
   visibile sul radargramma (a profondità diverse). "Termina" (o Invio). **Atteso:**
   compare il layer **"Radargram Lines"** (gruppo "Line Traces" nel pannello Livelli)
   con una feature; tabella attributi → `vertex_depths` contiene un valore diverso per
   ogni vertice, `depth_from`/`depth_to` coerenti col minimo/massimo cliccato,
   `z_source = radargram`, `profile_stem`/`channel` = profilo/canale correnti. La
   modalità resta attiva: si può disegnare subito un'altra linea.
2. **Punto**: attivare "Punto", un clic salva subito un punto nel layer
   **"Radargram Points"** con la profondità cliccata.
3. **Poligono**: attivare "Poligono", almeno 3 clic, poi "Termina": feature chiusa nel
   layer **"Radargram Polygons"**.
4. **Annulla vertice** (Backspace) toglie l'ultimo punto non ancora salvato; **Annulla**
   (Esc) scarta tutto il disegno in corso senza salvare nulla.
5. Cambiare profilo/canale **durante** un disegno non ancora terminato: **atteso** il
   disegno in corso viene scartato silenziosamente (nessuna geometria a cavallo di due
   profili).
6. In QGIS → menu plugin → **Build 3D** sul layer "Radargram Lines", modalità
   **"Per-vertex (from vertex_depths)"**: **atteso** la linea 3D risultante segue
   davvero la profondità disegnata punto per punto (non una Z costante come le altre
   modalità).
7. **[SUPERATO dal round 2 sotto]** "Misura velocità" a ricalcolo continuo — questo
   comportamento non esiste più: vedi la sezione "Persistenza reale, disegno
   visibile, velocità preimpostata" più in basso per il puntatore attuale (popup
   statico + "Calcola con iperbole" separato).
8. Attivare "Misura velocità"/"Calcola con iperbole" mentre "Linea"/"Punto"/"Poligono"
   è attivo (o viceversa): **atteso** si disattiva automaticamente l'altro, sono
   mutuamente esclusivi tra loro.

---

## Persistenza reale, disegno visibile, velocità preimpostata (2026-09-22, round 2, non committato)

Risposta ai 10 feedback dell'utente dopo l'uso reale del round precedente su SUASA
(screenshot del pannello con layer "Radargram Points/Lines" nel Layer panel e Trace
Info). Testato offscreen (`test_radargram_dock.py` 45, `test_radargram_capture.py` 8,
`test_velocity_store.py` 5, `test_hyperbola_math.py` 5,
`test_profile_viewer_interaction.py` 50 — 113 test totali tra i file toccati,
nessuna regressione sul resto della suite); **non** provato nella finestra reale di
QGIS. Serve un progetto GeoSurvey Studio collegato (Project
Manager) prima di disegnare — senza, un avviso lo dice esplicitamente invece di
salvare silenziosamente in memoria come prima.

1. **Persistenza reale (niente più layer temporanei)**: collegare un progetto,
   disegnare una linea sul radargramma. **Atteso:** compare
   `<progetto>/vector_layers/radargram_traces.gpkg` su disco. Chiudere QGIS
   **senza salvare il progetto** (o rimuovere e ricaricare il layer dal file), poi
   riaprire lo stesso `.gpkg` in QGIS: la feature c'è ancora — prova che non è più
   un memory layer. Disegnare poi anche un punto e un poligono: **atteso** tre
   layer ("Radargram Points/Lines/Polygons") tutti nello stesso `.gpkg`, nessuno
   sovrascrive gli altri.
2. **Disegno visibile sul radargramma**: dopo aver salvato una linea/punto/poligono,
   **atteso** che restino visibili (colore ciano) sul radargramma anche dopo aver
   disattivato "Linea"/ecc. Cambiare profilo e tornare indietro: **atteso** la
   geometria riappaia (letta dal `.gpkg`, non dalla sessione). Cambiare canale sullo
   stesso profilo: **atteso** sparisca (le geometrie sono per canale). Notare in
   pianta (vista mappa QGIS) che un poligono disegnato in profondità appare come una
   linea: è corretto, non un bug (est/nord derivano solo dalla posizione lungo il
   profilo).
3. **Attributi auto-compilati**: aprire la tabella attributi di "Radargram Lines".
   **Atteso** colonne `amplitude`, `velocity_m_ns`, `rdp` valorizzate (non vuote) per
   ogni feature disegnata da questo round in poi.
4. **Colonna "Radargramma" in Trace Info**: aprire il pannello Trace Info sul layer
   "Radargram Lines". **Atteso** una colonna "Radargramma" (accanto a "Time-slice")
   con `<variante> · <profilo> ch<canale>`; colonna "Vertices" non più sempre a 0 per
   queste feature (ora conta da `vertex_depths` quando manca il layer companion delle
   etichette).
5. **"Misura velocità" (popup statico)**: attivarla senza aver mai toccato nulla.
   **Atteso** un'etichetta gialla vicino al cursore con `v=0.100 m/ns εr=8.99 (di
   base)` che **non cambia** muovendo il mouse su punti diversi (a differenza del
   round 1).
6. **Campo velocità editabile**: cambiare il valore nel campo "Velocità" (es. a
   0,09). **Atteso**: si salva subito (nessun pulsante da premere), l'asse di
   profondità del radargramma nel pannello si ricalcola (confrontare la profondità
   massima mostrata prima/dopo), e riattivando "Misura velocità" il popup mostra il
   nuovo valore con `(manuale)`.
7. **"Calcola con iperbole" + salvataggio**: attivarla, cliccare l'apice di
   un'iperbole visibile, muovere il puntatore lungo un'ala. **Atteso** stessa lettura
   live del round 1 (ora in m/ns) e il pulsante "Salva come preimpostato" si abilita.
   Cliccarlo: **atteso** messaggio di conferma, e il campo "Velocità"/il popup
   "Misura velocità" riflettono subito il nuovo valore con `(da iperbole)`. Chiudere e
   riaprire il pannello (o QGIS): **atteso** il valore resta (letto da
   `velocity_overrides.json` nella cartella della variante).
8. **Profile Viewer, campo m/ns**: aprire il pannello Hyperbola Fit. **Atteso** un
   nuovo campo "velocity (m/ns)" accanto a RDP: cambiando l'uno si aggiorna l'altro
   in entrambe le direzioni.
9. **Profile Viewer, salvataggio velocità**: senza aver mai salvato una variante in
   questa sessione, premere "Salva velocità nella variante". **Atteso** messaggio
   "salva prima la variante". Salvare una variante (anche "Lavoro"), poi premere di
   nuovo: **atteso** conferma nello stato in basso. Aprire
   `GPR_Processing/processed/<variante>/velocity_overrides.json`: **atteso** il file
   c'è con `rdp`/`velocity_m_ns` coerenti col campo. Ripetere impostando un apice
   Hyperbola Fit e salvando una variante **senza** premere il pulsante: **atteso** il
   file compaia comunque (salvataggio automatico agganciato al salvataggio variante).
10. **Bug refresh Profile Viewer**: aprire un progetto con una variante già
    processata (filtri attivi). **Atteso** il radargramma appaia già filtrato al
    primo render, senza dover deflaggare/flaggare un filtro a mano.

---

## Correlazione radargramma <-> time-slice nel pannello (2026-09-23, non committato)

Risposta a 4 feedback dopo l'uso reale (il primo giro di questa correlazione,
uscito lo stesso giorno, non funzionava affatto). **Causa radice trovata ispezionando
il progetto SUASA reale dell'utente**: `_timeslice_layers_for_variant` (round 1)
richiedeva che la cartella delle time-slice avesse `processed/<variante>` come
antenato entro 4 livelli (`variant_dir_from_slice_dir`) - condizione MAI vera nel
progetto reale, dove le time-slice vivono sotto `SUASA/RadarDataExport/GeosurveyStudio/
GPR_Processing/timeslices/` mentre i profili processati stanno sotto
`SUASA/GeosurveyStudio/GPR_Processing/processed/<variante>/`: due alberi di cartelle
completamente separati. Inoltre lo slider di navigazione del dialogo principale
(`dlg.Dial`, vedi `catalog_group_mixin.update_visibility_with_dial`) esiste già e
naviga le time-slice per indice leggendo il **catalogo di progetto**
(`metadata/project_catalog.json`, `depth_from`/`depth_to` per raster), un meccanismo
diverso e più usato di `sliceset.json`.

**Fix**: la ricerca delle time-slice ora ignora del tutto la relazione di cartelle
con la variante (qualunque raster caricato nel progetto con una fascia di profondità
nota va bene, letta prima da `sliceset.json` e poi, se assente, dal catalogo), e il
pannello ora ascolta anche i cambi di visibilità nell'albero dei livelli (non solo i
propri clic), cosicché lo slider `Dial` del dialogo principale aggiorni la riga verde
in tempo reale. Testato offscreen (`test_radargram_dock.py`, 57 test, incluso un test
con time-slice in una cartella scollegata dalla variante e uno con dati solo nel
catalogo, senza `sliceset.json`); **non** ancora provato nella finestra reale di QGIS
sul progetto SUASA.

1. **Doppio clic sul radargramma -> attiva la time-slice**: con un gruppo di
   time-slice caricato nel canvas (Import to Canvas, o già presenti dal Project
   Manager/catalogo - non serve alcuna relazione con la variante aperta nel
   pannello), doppio clic sul radargramma a una profondità qualsiasi. **Atteso**: nel
   pannello Livelli resta spuntato solo il raster la cui fascia contiene quella
   profondità (le sorelle nello stesso gruppo dell'albero si spengono da sole), quel
   raster diventa il livello attivo, e la riga verde/fascia nel pannello si allinea
   alla sua finestra di profondità. Doppio clic più in profondità di tutte le fasce:
   **atteso** si attiva comunque la fascia più vicina (nessun errore).
2. **Slider di navigazione (`Dial`) -> riga verde live**: nel dialogo principale,
   selezionare il gruppo di time-slice e scorrere lo slider di navigazione (quello
   con tooltip "Navigation slider: switch visible rasters in selected groups").
   **Atteso**: con "Segui time-slice" attivo nel pannello Radargrammi processati, la
   riga verde/fascia seguono lo slider **senza** dover cliccare sul radargramma.
   Disattivando "Segui time-slice": **atteso** la riga verde smette di seguire lo
   slider.
3. **Clic su "Punta sulla mappa" -> indicatore preciso**: con una sola time-slice del
   gruppo spuntata/visibile, attivare "Punta sulla mappa" e cliccare un punto
   qualsiasi sopra l'area rilevata. **Atteso**: il pannello passa a profilo/canale/
   traccia più vicini (come prima) **e** la riga verde/fascia si aggiornano alla
   profondità della time-slice effettivamente visibile in quel momento sul canvas
   (non un vecchio riferimento rimasto da prima). Se nessuna time-slice è visibile,
   il riferimento precedente resta invariato (nessun azzeramento a caso).
4. **Nessuna time-slice caricata**: doppio clic sul radargramma senza aver mai
   caricato time-slice nel progetto. **Atteso** messaggio nello stato in basso
   ("Nessuna time-slice caricata nel progetto"), nessun errore/crash.
5. **"Misura velocità" ora si aggiorna dal vivo dopo un clic (round 4)**: chiarito
   con l'utente che il campo "Velocità" (spinbox accanto a "Materiale") deve restare
   il preimpostato statico per profilo, ma il **popup** "Misura velocità" deve invece
   mostrare un valore che cambia muovendo il puntatore, per confrontarlo con i
   valori tipici di materiale (aria/argilla/vuoto ecc. nel menu "Materiale..."), a
   supporto della lettura delle anomalie. Attivare "Misura velocità", **cliccare
   sull'apice di un'iperbole di diffrazione visibile** (il popup mostra "n/d" finche'
   il puntatore non e' sotto l'apice), poi muovere il puntatore lungo un'ala.
   **Atteso**: il popup mostra "v=... m/ns εr=... (dal vivo)" che cambia con la
   posizione del puntatore (stessa fisica di "Calcola con iperbole", ma letta dalla
   posizione del mouse invece che dalla rotellina), e una curva d'anteprima gialla
   segue il puntatore. **Esc** toglie l'apice e torna alla lettura statica
   preimpostata. Senza mai cliccare un apice, il popup resta la lettura statica
   preimpostata come prima (nessuna regressione per chi lo usa solo così). Cambiare
   profilo/canale o attivare "Calcola con iperbole" durante la misura: **atteso**
   l'apice si scarta silenziosamente.

## Annotazioni sul radargramma: classi di segnale, modifica, tempo (Fase 1, 2026-10-15)

Verificato solo offscreen (test automatici): l'aspetto reale dei controlli e la
fluidità del trascinamento vanno provati in QGIS.

1. **Classi e tasti 1-9**: nel pannello Radargrammi cliccare sul radargramma, premere
   `1`, `2`, `3`. **Atteso**: il menu "Segnale" cambia classe e si attiva da solo lo
   strumento giusto (punto per iperbole, linea per riflettore, poligono per
   attenuazione). Disegnare un'annotazione: compare nel colore della classe e si apre
   la scheda "Annotazione" sotto i comandi (classe, affidabilità, codice,
   interpretazione, note), **senza** dialogo modale.
2. **Modifica**: pulsante "Modifica". Clic su un'annotazione -> si seleziona (vertici
   come quadratini). Trascinare un vertice: si sposta e si salva al rilascio. Doppio
   clic su un segmento = nuovo vertice; clic destro su un vertice = lo toglie (mai
   sotto 1/2/3 vertici); `Canc` elimina; `1`-`9` cambiano la classe; `Esc` deseleziona.
   **Ctrl+Z / Ctrl+Y** (o i pulsanti ↶ ↷) annullano e ripetono, anche l'eliminazione.
   In QGIS aprire la tabella attributi di "Radargram Lines/Points/Polygons": le
   modifiche ci sono (campi nuovi `class_code`, `class_label`, `confidence`).
3. **Ancoraggio al segnale (tempo ns)**: disegnare una linea su un riflettore, poi
   cambiare "Velocità" (m/ns). **Atteso**: l'annotazione resta sul riflettore (la
   scala di profondità cambia, l'annotazione la segue). Le annotazioni fatte prima di
   questa versione non hanno il tempo e restano alla profondità salvata.
4. **⚙ Gestisci classi**: aggiungere/rinominare/cambiare colore; chiudere e riaprire il
   progetto -> le classi restano (`<progetto>/metadata/annotation_classes.json`).
5. Il vecchio dialogo "codice anomalia/note" dopo ogni disegno è ora spento
   (`trace_radargram_prompt_after_draw = False`): confermare che la scheda basta.

## Annotazioni sul radargramma, Fase 2 (2026-10-16)

Verificato solo offscreen (test automatici). Da provare in QGIS:

1. **Iperbole annotata**: strumento "Iperbole", clic sull'apice, rotellina fino a far
   combaciare la curva, poi **"Annota"**. **Atteso**: compare un punto giallo (classe
   «Iperbole di diffrazione») con la curva tratteggiata; la scheda mostra
   "v=… m/ns, εr=…, top a … m". In QGIS (tabella di "Radargram Points") `velocity_m_ns`
   e `rdp` sono quelli del fit, `source = iperbole_fit`, la profondità è quella del top
   dell'oggetto (non quella mostrata sul radargramma). Riaprendo il profilo la curva
   si ridisegna; cambiando la velocità del profilo resta sul segnale.
2. **Rettangolo**: premere `3` (attenuazione), poi "Rettangolo": due clic su angoli
   opposti salvano un poligono a 4 vertici. Premendo `3`/`4` restando in Rettangolo la
   modalità non cambia.
3. **Aggancia**: spuntarlo e cliccare vicino a un riflettore netto: il vertice si
   appoggia al picco (o ventre) di fase più vicino entro ~0,8 ns; vale anche
   trascinando un vertice in Modifica. **Maiusc+clic** in Linea/Poligono: i vertici fra
   il punto precedente e il nuovo seguono la fase del riflettore (max 40).
   Se il riflettore è debole/discontinuo il tracciamento può non seguirlo: va provato
   sui dati veri (SUASA) e, se serve, regolata la finestra (`_SNAP_NS`).
4. **Mostra**: il menu elenca le classi coi colori (è la legenda); togliere una classe
   o un livello di affidabilità la nasconde (e non si può più selezionare);
   "Etichette" mostra il codice anomalia vicino all'annotazione.
5. **Tooltip**: fermando il puntatore su un'annotazione (senza strumenti attivi, o in
   Modifica) compaiono classe, affidabilità, codice, interpretazione, note, profondità.
