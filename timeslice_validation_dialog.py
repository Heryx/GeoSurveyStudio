# -*- coding: utf-8 -*-
"""Compact UI helpers for time-slice import validation dialogs."""

import os
from collections import Counter

from PyQt5.QtWidgets import QMessageBox


def _record_name(rec):
    return os.path.basename(str((rec or {}).get("source_path") or ""))


def _compose_details(records):
    missing = []
    mismatch = []
    suspicious = []
    for rec in records or []:
        issues = rec.get("issues") if isinstance(rec.get("issues"), dict) else {}
        name = _record_name(rec)
        if issues.get("missing_crs"):
            missing.append(name)
        if issues.get("crs_mismatch"):
            mismatch.append(name)
        if issues.get("suspicious_extent"):
            suspicious.append(name)

    lines = []
    if missing:
        lines.append("Non-georeferenced images (missing CRS):")
        lines.extend([f"- {n}" for n in missing])
        lines.append("")
    if mismatch:
        lines.append("CRS mismatch images:")
        lines.extend([f"- {n}" for n in mismatch])
        lines.append("")
    if suspicious:
        lines.append("Suspicious extent images:")
        lines.extend([f"- {n}" for n in suspicious])
        lines.append("")

    if not lines:
        lines.append("No details available.")
    return "\n".join(lines).strip()


def _show_details(parent, details_text):
    QMessageBox.information(parent, "Import Details", details_text)


def _crs_summary(records):
    counter = Counter()
    for rec in records or []:
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        authid = str(meta.get("crs") or "").strip()
        if authid:
            counter[authid] += 1
    if not counter:
        return "None detected"
    items = [f"{k} ({v})" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(items)


def choose_timeslice_import_action(
    parent,
    records,
    issue_rows,
    missing_crs_count,
    mismatch_count,
    suspicious_count,
    project_crs_authid="",
    import_crs_authid="",
    scope_label="selected images",
):
    """
    Show compact validation dialog and return one of:
      - "cancel"
      - "set_crs" (user wants to choose the import CRS now, then re-validate)
      - "compatible" (import only CRS-compatible + non-suspicious)
      - "all_georef" (import all georeferenced, skip missing-CRS)
    """
    total = len(records or [])
    issues = len(issue_rows or [])
    if issues <= 0:
        return "all_georef"

    georef_count = max(0, total - int(missing_crs_count))
    crs_text = _crs_summary(records)
    proj = str(project_crs_authid or "").strip() or "Unknown"
    import_crs = str(import_crs_authid or "").strip()
    if int(mismatch_count) == 0 and georef_count > 0:
        project_status = "OK"
    elif georef_count <= 0:
        project_status = "No georeferenced images detected"
    else:
        project_status = "Mismatch detected"

    details_text = _compose_details(issue_rows or records)

    while True:
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Time-slice Import Check")
        msg.setText(f"Importing {total} image(s).")
        if int(missing_crs_count) > 0:
            if import_crs:
                recommended = (
                    f"Recommended action: Set Import CRS (currently {import_crs}) "
                    "if the images need a different one, then re-check."
                )
            else:
                recommended = (
                    "Recommended action: Set Import CRS. The images have a valid "
                    "position (world file) but no coordinate system; pick the CRS "
                    "of your survey (e.g. EPSG:32631) and it will be assigned on import."
                )
        else:
            recommended = "Recommended action: Import Compatible Only."

        msg.setInformativeText(
            "\n".join(
                [
                    f"Detected CRS: {crs_text}",
                    f"Project CRS: {proj}",
                    f"Import CRS: {import_crs or 'not set'}",
                    f"CRS check: {project_status}",
                    f"Non-georeferenced images: {int(missing_crs_count)} (will be skipped)",
                    f"Suspicious extent: {int(suspicious_count)}",
                    "",
                    recommended,
                ]
            )
        )

        details_btn = msg.addButton("View Details", QMessageBox.HelpRole)
        set_crs_btn = None
        if int(missing_crs_count) > 0:
            set_crs_btn = msg.addButton("Set Import CRS…", QMessageBox.ActionRole)
        compatible_btn = None
        all_georef_btn = None

        if int(mismatch_count) > 0 or int(suspicious_count) > 0:
            compatible_btn = msg.addButton("Import Compatible Only", QMessageBox.AcceptRole)
            all_georef_btn = msg.addButton("Import All Georeferenced", QMessageBox.ActionRole)
            msg.setDefaultButton(compatible_btn)
        else:
            all_georef_btn = msg.addButton("Continue Import", QMessageBox.AcceptRole)
            msg.setDefaultButton(all_georef_btn)

        cancel_btn = msg.addButton(QMessageBox.Cancel)
        msg.exec_()

        clicked = msg.clickedButton()
        if clicked == details_btn:
            _show_details(parent, details_text)
            continue
        if set_crs_btn is not None and clicked == set_crs_btn:
            return "set_crs"
        if clicked == cancel_btn:
            return "cancel"
        if compatible_btn is not None and clicked == compatible_btn:
            return "compatible"
        if all_georef_btn is not None and clicked == all_georef_btn:
            return "all_georef"
        return "cancel"
