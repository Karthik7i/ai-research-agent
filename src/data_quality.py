from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional

import pandas as pd


REQUIRED_FIELDS = (
    "symbol",
    "date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
)


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    symbol: Optional[str] = None
    date: Optional[date] = None


@dataclass
class QualityReport:
    checked: int
    valid: int
    invalid: int
    warnings: int
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _issue(
    code: str,
    severity: str,
    message: str,
    symbol: Optional[str] = None,
    record_date: Optional[date] = None,
) -> QualityIssue:
    return QualityIssue(code, severity, message, symbol, record_date)


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if not hasattr(result, "__len__") else False


def _normalize_symbol(value: Any) -> Optional[str]:
    if _is_null(value):
        return None
    symbol = str(value).strip().upper()
    return symbol or None


def _parse_date(value: Any) -> Optional[date]:
    if _is_null(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _report(checked: int, valid: int, issues: list[QualityIssue], metrics: Optional[Dict[str, Any]] = None) -> QualityReport:
    return QualityReport(
        checked=checked,
        valid=valid,
        invalid=checked - valid,
        warnings=sum(issue.severity == "warning" for issue in issues),
        issues=issues,
        metrics=metrics or {},
    )


def validate_market_record(
    record: Any,
    expected_symbol: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], QualityReport]:
    """Normalize and validate one market-data record without external side effects."""
    issues: list[QualityIssue] = []
    if not isinstance(record, Mapping):
        issues.append(_issue("invalid_record", "error", "Record must be a mapping."))
        return None, _report(1, 0, issues)

    missing_fields = [field for field in REQUIRED_FIELDS if field not in record or _is_null(record[field])]
    if missing_fields:
        issues.append(
            _issue(
                "missing_required_fields",
                "error",
                f"Missing required fields: {', '.join(missing_fields)}.",
            )
        )
        return None, _report(1, 0, issues)

    symbol = _normalize_symbol(record["symbol"])
    if symbol is None:
        issues.append(_issue("invalid_symbol", "error", "Symbol must be non-empty."))
        return None, _report(1, 0, issues)

    expected = _normalize_symbol(expected_symbol) if expected_symbol is not None else None
    if expected_symbol is not None and expected is None:
        issues.append(_issue("invalid_expected_symbol", "error", "Expected symbol must be non-empty."))
    elif expected is not None and symbol != expected:
        issues.append(
            _issue(
                "unexpected_symbol",
                "error",
                f"Record symbol {symbol} does not match expected symbol {expected}.",
                symbol,
            )
        )

    record_date = _parse_date(record["date"])
    if record_date is None:
        issues.append(_issue("invalid_date", "error", "Date must be ISO-formatted.", symbol))
    elif record_date > date.today():
        issues.append(_issue("future_date", "error", "Date cannot be in the future.", symbol, record_date))

    prices: Dict[str, Decimal] = {}
    for field_name in ("open_price", "high_price", "low_price", "close_price"):
        try:
            value = Decimal(str(record[field_name]))
        except (InvalidOperation, TypeError, ValueError):
            issues.append(_issue("invalid_ohlc", "error", f"{field_name} must be numeric.", symbol, record_date))
            continue
        if not value.is_finite():
            issues.append(_issue("non_finite_ohlc", "error", f"{field_name} must be finite.", symbol, record_date))
            continue
        prices[field_name] = value

    if len(prices) == 4:
        if prices["high_price"] < prices["low_price"]:
            issues.append(_issue("high_below_low", "error", "high_price must be at least low_price.", symbol, record_date))
        if not prices["low_price"] <= prices["open_price"] <= prices["high_price"]:
            issues.append(_issue("open_outside_range", "error", "open_price must be between low_price and high_price.", symbol, record_date))
        if not prices["low_price"] <= prices["close_price"] <= prices["high_price"]:
            issues.append(_issue("close_outside_range", "error", "close_price must be between low_price and high_price.", symbol, record_date))

    volume: Optional[int] = None
    try:
        volume_decimal = Decimal(str(record["volume"]))
        if not volume_decimal.is_finite():
            issues.append(_issue("invalid_volume", "error", "volume must be finite.", symbol, record_date))
        elif volume_decimal < 0:
            issues.append(_issue("negative_volume", "error", "volume must be non-negative.", symbol, record_date))
        elif volume_decimal != volume_decimal.to_integral_value():
            issues.append(_issue("non_integer_volume", "error", "volume must be integer-valued.", symbol, record_date))
        else:
            volume = int(volume_decimal)
    except (InvalidOperation, TypeError, ValueError):
        issues.append(_issue("invalid_volume", "error", "volume must be numeric.", symbol, record_date))

    if issues:
        return None, _report(1, 0, issues)

    normalized_record = {
        "symbol": symbol,
        "date": record_date,
        **prices,
        "volume": volume,
    }
    return normalized_record, _report(1, 1, issues)


def validate_market_dataframe(
    data: pd.DataFrame,
    min_history_rows: int = 21,
    suspicious_gap_days: int = 7,
) -> QualityReport:
    """Check in-memory market data quality without sorting or mutating the input."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if min_history_rows < 1:
        raise ValueError("min_history_rows must be positive")
    if suspicious_gap_days < 1:
        raise ValueError("suspicious_gap_days must be positive")

    checked = len(data)
    issues: list[QualityIssue] = []
    missing_columns = [column for column in REQUIRED_FIELDS if column not in data.columns]
    if missing_columns:
        issues.append(
            _issue(
                "missing_required_columns",
                "error",
                f"Missing required columns: {', '.join(missing_columns)}.",
            )
        )
        return _report(checked, 0, issues, {"missing_columns": missing_columns})

    frame = data.reset_index(drop=True)
    error_rows: set[int] = set()
    normalized_symbols: list[Optional[str]] = []
    parsed_dates: list[Optional[date]] = []

    for position, row in frame.iterrows():
        record = row.to_dict()
        symbol = _normalize_symbol(record["symbol"])
        record_date = _parse_date(record["date"])
        normalized_symbols.append(symbol)
        parsed_dates.append(record_date)

        for field_name in REQUIRED_FIELDS:
            if _is_null(record[field_name]):
                issues.append(_issue("null_value", "error", f"{field_name} cannot be null.", symbol, record_date))
                error_rows.add(position)

        _, record_report = validate_market_record(record)
        issues.extend(record_report.issues)
        if record_report.invalid:
            error_rows.add(position)

    key_frame = pd.DataFrame({"symbol": normalized_symbols, "date": parsed_dates})
    duplicate_mask = key_frame.notna().all(axis=1) & key_frame.duplicated(keep=False)
    for position in key_frame.index[duplicate_mask]:
        issues.append(
            _issue(
                "duplicate_symbol_date",
                "error",
                "Duplicate (symbol, date) row.",
                normalized_symbols[position],
                parsed_dates[position],
            )
        )
        error_rows.add(position)

    date_frame = pd.DataFrame({
        "symbol": normalized_symbols,
        "date": parsed_dates,
        "position": range(checked),
    }).dropna(subset=["symbol", "date"])
    history_rows_by_symbol: Dict[str, int] = {}
    for symbol, group in date_frame.groupby("symbol", sort=True):
        history_rows_by_symbol[symbol] = len(group)
        if not group["date"].is_monotonic_increasing:
            issues.append(
                _issue(
                    "out_of_order_dates",
                    "warning",
                    "Rows are not in chronological order for this symbol.",
                    symbol,
                )
            )

        if len(group) < min_history_rows:
            issues.append(
                _issue(
                    "insufficient_history",
                    "warning",
                    f"{len(group)} rows available; {min_history_rows} required.",
                    symbol,
                )
            )

        ordered_dates = group.sort_values("date")["date"].tolist()
        for previous_date, next_date in zip(ordered_dates, ordered_dates[1:]):
            if (next_date - previous_date).days > suspicious_gap_days:
                issues.append(
                    _issue(
                        "suspicious_gap",
                        "warning",
                        f"Gap of {(next_date - previous_date).days} calendar days.",
                        symbol,
                        next_date,
                    )
                )

    valid = checked - len(error_rows)
    metrics = {
        "symbols": len(history_rows_by_symbol),
        "history_rows_by_symbol": history_rows_by_symbol,
        "duplicate_rows": int(duplicate_mask.sum()),
        "min_history_rows": min_history_rows,
        "suspicious_gap_days": suspicious_gap_days,
    }
    return _report(checked, valid, issues, metrics)


def validate_training_readiness(
    data: pd.DataFrame,
    min_history_rows: int = 21,
) -> QualityReport:
    """Report whether each symbol has enough raw data for training preparation."""
    report = validate_market_dataframe(data, min_history_rows=min_history_rows)
    history_rows_by_symbol = report.metrics.get("history_rows_by_symbol", {})
    eligible_symbols = sorted(
        symbol for symbol, count in history_rows_by_symbol.items() if count >= min_history_rows
    )
    ineligible_symbols = sorted(
        symbol for symbol, count in history_rows_by_symbol.items() if count < min_history_rows
    )
    eligible_row_count = sum(history_rows_by_symbol[symbol] for symbol in eligible_symbols)
    issues = list(report.issues)

    if not eligible_symbols:
        issues.append(
            _issue("no_eligible_symbols", "error", "No symbols meet the history requirement."))
    if eligible_row_count == 0:
        issues.append(
            _issue("empty_eligible_dataset", "error", "No rows remain after eligibility filtering."))

    metrics = dict(report.metrics)
    metrics.update({
        "eligible_symbols": eligible_symbols,
        "ineligible_symbols": ineligible_symbols,
        "eligible_symbol_count": len(eligible_symbols),
        "has_eligible_symbols": bool(eligible_symbols),
        "eligible_row_count": eligible_row_count,
        "dataset_empty_after_eligibility_filter": eligible_row_count == 0,
    })
    return _report(report.checked, report.valid, issues, metrics)
