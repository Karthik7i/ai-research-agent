from datetime import date, timedelta

import pandas as pd
import pytest

from src.data_quality import (
    validate_market_dataframe,
    validate_market_record,
    validate_training_readiness,
)


def market_record(**overrides):
    record = {
        "symbol": " aapl ",
        "date": "2026-08-09",
        "open_price": "220.00",
        "high_price": "225.00",
        "low_price": "218.00",
        "close_price": "223.50",
        "volume": "1000000",
    }
    record.update(overrides)
    return record


def market_frame(symbol="IBM", rows=21, start="2026-01-01"):
    dates = pd.date_range(start, periods=rows)
    return pd.DataFrame({
        "symbol": [symbol] * rows,
        "date": dates,
        "open_price": [100.0] * rows,
        "high_price": [105.0] * rows,
        "low_price": [95.0] * rows,
        "close_price": [102.0] * rows,
        "volume": [1000] * rows,
    })


def issue_codes(report):
    return [issue.code for issue in report.issues]


def test_valid_record_is_normalized():
    normalized, report = validate_market_record(market_record(), expected_symbol="aapl")

    assert report.valid == 1
    assert report.invalid == 0
    assert normalized["symbol"] == "AAPL"
    assert normalized["date"] == date(2026, 8, 9)
    assert normalized["volume"] == 1000000


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"high_price": "217"}, "high_below_low"),
        ({"open_price": "217"}, "open_outside_range"),
        ({"open_price": "226"}, "open_outside_range"),
        ({"close_price": "217"}, "close_outside_range"),
        ({"close_price": "226"}, "close_outside_range"),
    ],
)
def test_record_rejects_each_ohlc_boundary_violation(overrides, code):
    _, report = validate_market_record(market_record(**overrides))

    assert report.invalid == 1
    assert code in issue_codes(report)


def test_record_rejects_missing_fields():
    record = market_record()
    del record["volume"]

    _, report = validate_market_record(record)

    assert report.invalid == 1
    assert "missing_required_fields" in issue_codes(report)


@pytest.mark.parametrize("value", ["not-a-date", date.today() + timedelta(days=1)])
def test_record_rejects_malformed_and_future_dates(value):
    _, report = validate_market_record(market_record(date=value))

    assert report.invalid == 1
    assert {"invalid_date", "future_date"} & set(issue_codes(report))


@pytest.mark.parametrize(
    ("volume", "is_valid", "code"),
    [
        ("0", True, None),
        ("10.0", True, None),
        ("1.5", False, "non_integer_volume"),
        ("-1", False, "negative_volume"),
        ("NaN", False, "invalid_volume"),
    ],
)
def test_record_validates_volume_edge_cases(volume, is_valid, code):
    _, report = validate_market_record(market_record(volume=volume))

    assert bool(report.valid) is is_valid
    if code:
        assert code in issue_codes(report)


def test_record_normalizes_symbols_and_rejects_unexpected_symbols():
    normalized, valid_report = validate_market_record(market_record(), expected_symbol=" AAPL ")
    _, invalid_report = validate_market_record(market_record(), expected_symbol="MSFT")

    assert normalized["symbol"] == "AAPL"
    assert valid_report.valid == 1
    assert "unexpected_symbol" in issue_codes(invalid_report)


def test_dataframe_reports_duplicate_rows_and_nulls():
    data = pd.concat([market_frame(rows=21), market_frame(rows=1)]).reset_index(drop=True)
    data.loc[2, "volume"] = None

    report = validate_market_dataframe(data)

    assert "duplicate_symbol_date" in issue_codes(report)
    assert "null_value" in issue_codes(report)
    assert report.metrics["duplicate_rows"] == 2
    assert report.invalid >= 2


def test_dataframe_reports_insufficient_history_at_twenty_rows():
    report = validate_market_dataframe(market_frame(rows=20))

    assert "insufficient_history" in issue_codes(report)
    assert report.metrics["history_rows_by_symbol"] == {"IBM": 20}


def test_dataframe_accepts_minimum_history_at_twenty_one_rows():
    report = validate_market_dataframe(market_frame(rows=21))

    assert "insufficient_history" not in issue_codes(report)
    assert report.valid == 21


def test_weekend_gap_is_not_suspicious():
    data = market_frame(rows=2, start="2026-01-02")
    data.loc[1, "date"] = pd.Timestamp("2026-01-05")

    report = validate_market_dataframe(data, min_history_rows=1, suspicious_gap_days=7)

    assert "suspicious_gap" not in issue_codes(report)


def test_gap_over_seven_days_is_a_warning():
    data = market_frame(rows=2, start="2026-01-01")
    data.loc[1, "date"] = pd.Timestamp("2026-01-10")

    report = validate_market_dataframe(data, min_history_rows=1, suspicious_gap_days=7)

    assert "suspicious_gap" in issue_codes(report)
    assert report.warnings >= 1


def test_training_readiness_identifies_mixed_eligible_and_ineligible_symbols():
    eligible = market_frame(symbol="IBM", rows=21)
    ineligible = market_frame(symbol="AAPL", rows=20)
    report = validate_training_readiness(pd.concat([eligible, ineligible]), min_history_rows=21)

    assert report.metrics["eligible_symbols"] == ["IBM"]
    assert report.metrics["ineligible_symbols"] == ["AAPL"]
    assert report.metrics["has_eligible_symbols"] is True
    assert report.metrics["dataset_empty_after_eligibility_filter"] is False


def test_training_readiness_reports_no_eligible_symbols():
    report = validate_training_readiness(market_frame(rows=20), min_history_rows=21)

    assert report.metrics["eligible_symbols"] == []
    assert report.metrics["has_eligible_symbols"] is False
    assert report.metrics["dataset_empty_after_eligibility_filter"] is True
    assert {"no_eligible_symbols", "empty_eligible_dataset"} <= set(issue_codes(report))
