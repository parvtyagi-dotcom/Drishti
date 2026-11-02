import pytest

from backend.modules.anpr.normalize import MAX_FORMAT_FIXES, clean_plate_text, correct_plate_format
from backend.modules.anpr.ocr import normalize_plate


@pytest.mark.parametrize("plate", [
    "MH12AB1234", "DL3CAB1234", "DL8CAF1234", "UP16BT5555", "HR26DK8337",
    "DL1C1234", "DL011234", "21BH1234AB", "22BH0001A",
])
def test_correct_plates_are_left_alone(plate):
    assert correct_plate_format(plate) == plate


@pytest.mark.parametrize("read, plate", [
    ("MH12A81234", "MH12AB1234"),   # 8 read for B in the series
    ("MH12AB1Z34", "MH12AB1234"),   # Z read for 2 in the number
    ("KA05MZ2O19", "KA05MZ2019"),   # O read for 0 in the number
    ("DL1CI234", "DL1C1234"),       # I read for 1 in the number
    ("0D02AB1234", "OD02AB1234"),   # 0 read for O in the state code
    ("MH12ABI2S4", "MH12AB1254"),   # two fixes is still within the cap
    ("2IBH1234A8", "21BH1234AB"),   # Bharat series: year and series letters
])
def test_letter_digit_swaps_are_fixed_by_position(read, plate):
    assert correct_plate_format(read) == plate


@pytest.mark.parametrize("read", [
    "TEST123",      # the demo clip's plate: no Indian layout fits
    "XZY575",       # too short for any layout
    "ABCDEFGH",     # no digits where the number must be
    "ZZ12AB1234",   # fits the shape, but ZZ is no state code
    "MH12AB12C4",   # C is not a look-alike of any digit
    "",
])
def test_reads_that_are_not_indian_plates_pass_through(read):
    assert correct_plate_format(read) == read


def test_too_many_fixes_is_treated_as_a_guess_and_left_alone():
    read = "MHI2A8I234"  # would need I->1, 8->B, I->1
    assert MAX_FORMAT_FIXES == 2
    assert correct_plate_format(read) == read


def test_bharat_series_needs_the_literal_bh():
    assert correct_plate_format("21BX1234AB") == "21BX1234AB"


def test_clean_plate_text_fixes_a_misread_state_code_before_trimming():
    # Trimming alone would cut "0D..." down looking for a state code.
    assert clean_plate_text("0D 02 AB 12O4", normalize_plate) == "OD02AB1204"


def test_clean_plate_text_fixes_after_dropping_leading_junk():
    assert clean_plate_text("X MH12A81234", normalize_plate) == "MH12AB1234"


def test_clean_plate_text_merges_variant_reads_of_one_plate():
    # The aggregator groups by exact text, so variant reads of one plate
    # must come out identical to count as agreeing observations.
    reads = ["MH12AB1234", "MH12A81234", "MH12AB1Z34", "MHI2AB1234"]
    assert {clean_plate_text(r, normalize_plate) for r in reads} == {"MH12AB1234"}
