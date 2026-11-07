"""Esporta la scena della vista 3D in KML/KMZ (Google Earth, QGIS, ...).

Cosa entra: le tracce dei profili GPR e delle sezioni ERT come linee con il loro nome, e ogni time-slice
come ``GroundOverlay`` georiferito (anche ruotato: ``gx:LatLonQuad``). Il KML e' un formato di mappa: la
profondita' non c'e'. La slice sta sul terreno e la sua fascia di profondita' e' scritta nel nome e nella
descrizione; i radargrammi verticali restano nella vista 3D (si esportano come immagine).
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from xml.sax.saxutils import escape

import numpy as np

_KML_NS = 'xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2"'


def _coords(xy, to_lonlat) -> str:
    pts = []
    for x, y in np.asarray(xy, dtype=np.float64):
        lon, lat = to_lonlat(float(x), float(y))
        pts.append(f"{lon:.8f},{lat:.8f},0")
    return " ".join(pts)


def _line_folder(title: str, items, to_lonlat, color: str, describe) -> str:
    if not items:
        return ""
    out = [f"<Folder><name>{escape(title)}</name>"]
    for it in items:
        out.append(
            f"<Placemark><name>{escape(str(it['name']))}</name>"
            f"<description>{escape(describe(it))}</description>"
            f"<Style><LineStyle><color>{color}</color><width>3</width></LineStyle></Style>"
            f"<LineString><tessellate>1</tessellate><coordinates>{_coords(it['xy'], to_lonlat)}</coordinates>"
            "</LineString></Placemark>")
    out.append("</Folder>")
    return "".join(out)


def _slice_png(it, path: str) -> None:
    from matplotlib import image as mpimg

    mpimg.imsave(path, np.asarray(it["rgba"], dtype=np.uint8), format="png")


def _safe(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
    return keep or "slice"


def write_kml(path: str, scene: dict, to_lonlat, title: str = "Vista 3D profili", slice_opacity: float = 0.85) -> dict:
    """Scrive ``path`` (.kml o .kmz). ``scene`` = {"gprs": [...], "erts": [...], "slices": [...]}.

    Voci GPR/ERT: ``name``, ``xy`` (N,2) nel CRS della scena. Slice: ``name``, ``corners`` (4,2) in ordine
    alto-sinistra, alto-destra, basso-destra, basso-sinistra, ``rgba`` (h,w,4) uint8, ``z_from``, ``z_to``.
    ``to_lonlat(x, y) -> (lon, lat)``. Restituisce i conteggi scritti.
    """
    path = str(path)
    kmz = path.lower().endswith(".kmz")
    gprs, erts, slices = scene.get("gprs") or [], scene.get("erts") or [], scene.get("slices") or []
    if not (gprs or erts or slices):
        raise ValueError("la scena e' vuota: niente da esportare")
    stem = _safe(os.path.splitext(os.path.basename(path))[0])
    img_dir = "images" if kmz else f"{stem}_slices"
    alpha = int(round(max(0.0, min(1.0, float(slice_opacity))) * 255))

    parts = [f'<?xml version="1.0" encoding="UTF-8"?><kml {_KML_NS}><Document><name>{escape(title)}</name>']
    parts.append(_line_folder("Profili GPR", gprs, to_lonlat, "ff0000ff",
                              lambda it: f"Profilo GPR, profondità {float(it.get('depth', 0.0)):.2f} m"))
    parts.append(_line_folder("Sezioni ERT", erts, to_lonlat, "ffff7f00", lambda it: "Sezione ERT"))
    images = []                                   # (nome nel pacchetto, immagine rgba)
    if slices:
        parts.append("<Folder><name>Time-slice</name>")
        for i, it in enumerate(slices):
            c = np.asarray(it["corners"], dtype=np.float64)
            tl, tr, br, bl = c[0], c[1], c[2], c[3]
            quad = " ".join(_coords([p], to_lonlat) for p in (bl, br, tr, tl))      # antiorario dal basso-sinistra
            rel = f"{img_dir}/{i:03d}_{_safe(it['name'])}.png"
            images.append((rel, it))
            z0, z1 = float(it.get("z_from", 0.0)), float(it.get("z_to", 0.0))
            label = f"{z0:.2f}–{z1:.2f} m  ({it['name']})"
            parts.append(
                f"<GroundOverlay><name>{escape(label)}</name>"
                f"<description>Time-slice a profondità {z0:.2f}–{z1:.2f} m (nel KML sta sul terreno).</description>"
                f"<color>{alpha:02x}ffffff</color><Icon><href>{escape(rel)}</href></Icon>"
                f"<gx:LatLonQuad><coordinates>{quad}</coordinates></gx:LatLonQuad></GroundOverlay>")
        parts.append("</Folder>")
    parts.append("</Document></kml>")
    kml = "".join(parts)

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    if kmz:
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("doc.kml", kml)
            for rel, it in images:
                png = os.path.join(tmp, os.path.basename(rel))
                _slice_png(it, png)
                zf.write(png, rel)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(kml)
        if images:
            os.makedirs(os.path.join(folder, img_dir), exist_ok=True)
            for rel, it in images:
                _slice_png(it, os.path.join(folder, rel))
    return {"gpr": len(gprs), "ert": len(erts), "slice": len(slices)}
