"""Naive, seasonal naive, AMC-3 and eLMIS AMC forecasts.

Each function takes the clean panel and an origin month index and returns only
what was known at that origin. Series with no usable history get no forecast.
"""
from src.io import SERIES_KEYS

BASELINES = ["naive", "seasonal_naive", "amc_3", "elmis_amc"]


def _valid_history(panel, origin):
    history = panel[(panel["month_index"] <= origin) & panel["y"].notna()]
    return history.sort_values(SERIES_KEYS + ["month_index"])


def naive(panel, origin):
    """Last valid value per series."""
    return _valid_history(panel, origin).groupby(SERIES_KEYS)["y"].last()


def amc_3(panel, origin):
    """Mean of the last 3 valid months, or of however many exist if fewer."""
    history = _valid_history(panel, origin)
    return history.groupby(SERIES_KEYS).tail(3).groupby(SERIES_KEYS)["y"].mean()


def elmis_amc(panel, origin):
    """The AMC the system recorded at the origin. Absent if the site did not report."""
    at_origin = panel[(panel["month_index"] == origin) & panel["reported"]]
    return at_origin.set_index(SERIES_KEYS)["average_monthly_consumption"]


def seasonal_naive(panel, origin, target):
    """Valid value 12 months before the target month, falling back to naive."""
    year_ago = _valid_history(panel, origin)
    year_ago = year_ago[year_ago["month_index"] == target - 12]
    return year_ago.set_index(SERIES_KEYS)["y"].combine_first(naive(panel, origin))
