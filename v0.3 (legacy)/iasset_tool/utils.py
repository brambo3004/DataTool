\
"""
Algemene hulpfuncties.

Deze functies bevatten geen Streamlit-code en zijn daardoor eenvoudig
automatisch te testen.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def is_empty_value(value: Any) -> bool:
    """
    Bepaal of een waarde inhoudelijk leeg is.

    Waarom apart?
    iASSET-exports kunnen lege cellen, NaN, None, spaties of de tekst "nan"
    bevatten. Voor de domeinlogica willen we die allemaal hetzelfde behandelen.
    """
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "<na>"}


def clean_display_value(value: Any) -> str:
    """
    Maak een waarde netjes voor scherm en export.

    Voorbeeld:
    - 2005.0 wordt 2005
    - NaN wordt lege tekst
    - N351-HRB-01.2 blijft intact, omdat daar letters in staan
    """
    if is_empty_value(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0") and not any(char.isalpha() for char in text):
        return text[:-2]

    return text


def normalize_text(value: Any) -> str:
    """
    Maak tekst geschikt voor vergelijkingen in de code.

    Voor domeinwaarden zoals subthema's willen we niet afhankelijk zijn van
    hoofdletters of spaties aan begin/einde.
    """
    if is_empty_value(value):
        return ""
    return str(value).strip().lower()


def parse_hm_sort(value: Any, fallback: float = 99999.9) -> float:
    """
    Zet een metrering/hectometrering om naar een sorteerbaar getal.

    De app gebruikt de waarde alleen voor sortering. Daarom is een veilige
    fallback beter dan crashen op een afwijkende iASSET-waarde.
    """
    if is_empty_value(value):
        return fallback

    text = str(value).strip().replace(",", ".")
    number = pd.to_numeric(text, errors="coerce")

    if pd.isna(number):
        return fallback

    try:
        return float(number)
    except (TypeError, ValueError, OverflowError):
        return fallback


def parse_date_info(value: Any) -> tuple[int, int]:
    """
    Haal jaar en maand uit een datumachtige waarde.

    De iASSET-export kan jaartallen, datums of lege waarden bevatten.
    We geven altijd een tuple terug: (jaar, maand). Bij onbekend wordt dit (0, 0).
    """
    if is_empty_value(value):
        return 0, 0

    text = clean_display_value(value)

    if len(text) == 4 and text.isdigit():
        return int(text), 0

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return int(parsed.year), int(parsed.month)

    # Fallback: soms begint een tekst alsnog met een jaartal.
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4]), 0

    return 0, 0


def safe_int(value: Any) -> int | None:
    """
    Zet een waarde veilig om naar int.

    Wordt gebruikt voor IDs uit autosave-csv's. CSV maakt van 12 soms "12.0".
    """
    if is_empty_value(value):
        return None

    try:
        number = float(str(value).strip())
    except ValueError:
        return None

    if math.isnan(number):
        return None

    return int(number)
