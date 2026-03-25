import re
from psychopy import event, core


def is_valid_name(name: str) -> bool:
    name = name.strip()
    pattern = r"^[\w\s\-']+$"
    return bool(re.match(pattern, name, re.UNICODE))

is_valid_number_map = {
    'int': lambda v, lo, hi: _check_int(v, lo, hi),
    'float': lambda v, lo, hi: _check_float(v, lo, hi)
}


def _check_int(val, lo=None, hi=None):
    try:
        i = int(val)
    except (ValueError, TypeError):
        return False
    return (lo is None or i >= lo) and (hi is None or i <= hi)


def _check_float(val, lo=None, hi=None):
    try:
        f = float(val)
    except (ValueError, TypeError):
        return False
    return (lo is None or f >= lo) and (hi is None or f <= hi)


def is_valid_number(val, type='int', lo=None, hi=None):
    return is_valid_number_map[type](val, lo, hi)