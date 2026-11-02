"""India-aware plate-text cleanup, layered on top of ocr.py's basic
normalize_plate() (uppercase + strip formatting).

The state-code-trim heuristic is adapted from the reference project
(RonLek/ALPR-and-Identification-for-Indian-Vehicles, ocr.py `resultplate()`):
Indian plates always start with a real 2-letter state code, and OCR output is
often preceded by junk (a partial "INDIA" watermark, a smudge misread as a
character) -- trimming from the left until a real state code appears cleans
that up. Unlike the reference project, a failed trim falls back to the
untrimmed text rather than discarding the read entirely.
"""

from __future__ import annotations

INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "CH", "DD", "DL", "JK", "LA", "LD", "PY",
}

_WATERMARK_NOISE = ("INDIA", "BHARAT")


def strip_watermark_noise(text: str) -> str:
    """Drops common plate-sticker watermark text ('INDIA', 'BHARAT') that OCR
    sometimes picks up alongside the actual registration number. Deliberately
    does NOT strip plain 'IN'/'IND' substrings -- those are too short and
    would eat real plate characters (e.g. a plate ending in ...IN12)."""
    upper = text.upper()
    for token in _WATERMARK_NOISE:
        upper = upper.replace(token, "")
    return upper


def trim_to_state_code(normalized_text: str) -> str:
    """Trims leading characters until the first two match a real Indian state
    code, exactly as the reference project's resultplate() does. Returns the
    input UNCHANGED if no state code is ever found (never discards a read)."""
    text = normalized_text
    while len(text) >= 2 and text[:2] not in INDIAN_STATE_CODES:
        text = text[1:]
    return text if len(text) >= 2 else normalized_text


# Characters OCR swaps between the two classes on plate fonts. A position
# that must be a digit gets the left column's letter turned into the digit,
# and the other way round. Only shape-alike pairs: a C or an X in a digit
# slot is not a misread 0 or 8, it is a different plate, so it is left alone.
_LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1",
                    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G",
                    "7": "T", "8": "B"}

# More fixes than this and the read is more guess than plate.
MAX_FORMAT_FIXES = 2


def _layouts(n: int):
    """Every way a plate of length n fits an Indian registration format, as
    (classes, fixed): classes is one 'L'/'D' per character, fixed maps a
    position to the only character allowed there.

    Standard:      SS 0-99 [A-Z]{0,3} 0000  e.g. DL3CAB1234, MH12AB1234, DL011234
    Bharat series: YY BH 0000 [A-Z]{1,2}    e.g. 21BH1234AB
    The unique number is always written with 4 digits (zero-padded), so the
    shorter-number case is not a layout here - it simply is not corrected.
    """
    for rto in (1, 2):
        series = n - 2 - rto - 4
        if 0 <= series <= 3:
            yield "LL" + "D" * rto + "L" * series + "DDDD", {}
    letters = n - 8
    if 1 <= letters <= 2:
        yield "DDLLDDDD" + "L" * letters, {2: "B", 3: "H"}


def correct_plate_format(text: str) -> str:
    """Fixes OCR letter/digit swaps by position, when the read fits exactly
    one Indian plate layout: 'MH12A81234' -> 'MH12AB1234', '0D02AB12O4' ->
    'OD02AB1204'. A standard plate must also start with a real state code.

    Conservative on purpose: if no layout fits within MAX_FORMAT_FIXES
    substitutions, or two layouts fit equally well but read differently, the
    text comes back unchanged. A read that is not an Indian plate at all (a
    foreign or a test plate) therefore passes through untouched."""
    if not text:
        return text
    best: dict[str, int] = {}
    for classes, fixed in _layouts(len(text)):
        out, fixes = [], 0
        for i, (ch, cls) in enumerate(zip(text, classes)):
            if cls == "D" and not ch.isdigit():
                ch, fixes = _LETTER_TO_DIGIT.get(ch), fixes + 1
            elif cls == "L" and ch.isdigit():
                ch, fixes = _DIGIT_TO_LETTER.get(ch), fixes + 1
            if ch is None or (i in fixed and ch != fixed[i]):
                break
            out.append(ch)
        else:
            candidate = "".join(out)
            if not fixed and candidate[:2] not in INDIAN_STATE_CODES:
                continue
            if fixes <= MAX_FORMAT_FIXES:
                best[candidate] = min(fixes, best.get(candidate, fixes))
    if not best:
        return text
    fewest = min(best.values())
    winners = [c for c, f in best.items() if f == fewest]
    return winners[0] if len(winners) == 1 else text


def clean_plate_text(raw: str, normalize_fn) -> str:
    """Full India-aware cleanup pass: strip watermark tokens, run the
    project's existing normalize_fn (ocr.normalize_plate -- uppercase +
    whitelist), fix letter/digit swaps by plate layout, and trim leading junk
    to the state code. Takes normalize_fn as a parameter rather than
    importing it, to avoid a circular import between normalize.py and ocr.py.

    The layout fix runs on the whole read first, because the trim would
    otherwise eat a misread state code ('0D...' for 'OD...'), and again after
    the trim, for a read that only fits once its leading junk is gone."""
    if not raw:
        return ""
    text = strip_watermark_noise(raw)
    text = normalize_fn(text)
    if not text:
        return text
    fixed = correct_plate_format(text)
    if fixed != text or text[:2] in INDIAN_STATE_CODES:
        return trim_to_state_code(fixed)
    return correct_plate_format(trim_to_state_code(text))
