# Proposta di comunicazione — GEOFISIC-ARQ

> Requisiti rispettati: titolo, autori, parole chiave (max 5), abstract (max 300 parole). Versioni in inglese e castigliano. Sostituisci co-autori e affiliazioni dove indicato tra parentesi quadre.

---

## English version

**Title**
Browsing hundreds of time-slices: a QGIS plugin for the integrated management, processing and comparison of GPR depth-slices

**Author(s)**
Giuseppe Guarino [+ co-authors] — Alma Mater Studiorum, Università di Bologna [+ affiliations]

**Keywords**
GPR; time-slices; QGIS; open-source software; archaeological prospection

**Abstract** (≈270 words)

Ground-penetrating radar (GPR) surveys of archaeological sites routinely generate very large sets of depth/time-slices: a single area, acquired with multi-channel arrays and re-processed under different parameters, can yield hundreds of georeferenced raster images. Their interpretation is often hampered less by data quality than by the practical difficulty of organising, navigating and comparing so many outputs within a coherent spatial framework. We present an open-source QGIS plugin conceived to address this bottleneck. The plugin integrates the GPR time-slice workflow inside a single GIS environment: it imports already-processed slices, catalogues them into depth-ordered groups, and links each slice to its depth and to the corresponding radargram profile, enabling rapid browsing through a depth slider and side-by-side inspection. It also supports processing GPR data directly in QGIS — reading radar volumes, applying a configurable trace pipeline (dewow, time-zero, background removal, band-pass, gain/AGC, Hilbert envelope) and generating georeferenced slices by IDW interpolation from point-cloud (LAS/COPC) or tabular inputs. Processed datasets can be saved as named variants (for example with and without band-pass), so that alternative interpretations can be produced and compared reproducibly. The main benefit is a faster, better-documented consultation of hundreds of slices without leaving the GIS used for archaeological interpretation. We also state current limitations honestly: automatic parameter tuning is not yet reliable (notably band-pass estimation), support for proprietary acquisition formats has been partly reverse-engineered and needs validation across instruments and versions, performance on very large volumes is still being optimised, and the tool lacks a formal, quantitative benchmark against established software. We present these as open problems for community discussion.

---

## Versión en castellano

**Título**
Consultar cientos de time-slices: un complemento de QGIS para la gestión, el procesado y la comparación integrados de secciones de profundidad de georradar

**Autor/es**
Giuseppe Guarino [+ coautores] — Alma Mater Studiorum, Università di Bologna [+ afiliaciones]

**Palabras clave**
georradar (GPR); time-slices; QGIS; software libre; prospección arqueológica

**Resumen** (≈270 palabras)

Las prospecciones con georradar (GPR) en yacimientos arqueológicos generan de forma habitual conjuntos muy grandes de secciones de profundidad (time-slices): una sola área, adquirida con antenas multicanal y reprocesada con distintos parámetros, puede producir cientos de imágenes ráster georreferenciadas. Su interpretación se ve dificultada, más que por la calidad del dato, por la dificultad práctica de organizar, navegar y comparar tantos resultados dentro de un marco espacial coherente. Presentamos un complemento (plugin) de QGIS, de código abierto, concebido para resolver este cuello de botella. El complemento integra el flujo de trabajo de las time-slices dentro de un único entorno SIG: importa secciones ya procesadas, las cataloga en grupos ordenados por profundidad y vincula cada sección con su profundidad y con el radargrama correspondiente, permitiendo una consulta rápida mediante un deslizador de profundidad y la inspección comparada. También permite procesar el dato GPR directamente en QGIS —lectura de volúmenes radar, aplicación de una cadena de filtros configurable (dewow, tiempo cero, eliminación de fondo, paso-banda, ganancia/AGC, envolvente de Hilbert) y generación de secciones georreferenciadas por interpolación IDW a partir de nubes de puntos (LAS/COPC) o datos tabulares. Los conjuntos procesados pueden guardarse como variantes con nombre (por ejemplo, con y sin paso-banda), de modo que las interpretaciones alternativas se produzcan y comparen de forma reproducible. El beneficio principal es una consulta más rápida y mejor documentada de cientos de secciones sin salir del SIG empleado en la interpretación arqueológica. Exponemos también, con honestidad, las limitaciones actuales: el ajuste automático de parámetros aún no es fiable (en particular la estimación del paso-banda), el soporte de formatos propietarios se ha obtenido en parte por ingeniería inversa y requiere validación entre instrumentos y versiones, el rendimiento en volúmenes muy grandes se está optimizando, y falta un banco de pruebas cuantitativo frente a programas consolidados. Las presentamos como problemas abiertos para el debate con la comunidad.
