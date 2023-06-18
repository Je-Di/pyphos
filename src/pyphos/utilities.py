"""
Utility functions for using photonic modules.
"""
import math


# pylint: disable=invalid-name
def db_to_factor(db: dict[str:float] | float) -> dict[str:float] | float:
    """Convert single value or list from dB to factor.

    Arguments:
        db -- The dB value(s) to be converted

    Returns:
        The input converted to (a) factor(s)
    """
    try:
        iterator = iter(db)
    except TypeError:
        factor = 10 ** (db / 10)
    else:
        factor = dict()
        for loss_type in iterator:
            factor[loss_type] = 10 ** (db[loss_type] / 10)
    return factor


def factor_to_db(factor: dict[str:float] | float) -> dict[str:float] | float:
    """Convert single value or list from factor to dB.

    Arguments:
        factor -- The factor(s) to be converted

    Returns:
        The input converted to dB
    """
    try:
        iterator = iter(factor)
    except TypeError:
        db = 10 * math.log10(factor)
    else:
        db = dict()
        for loss_type in iterator:
            db[loss_type] = 10 * math.log10(factor[loss_type])
    return db
