
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    OFFICIAL = "official"
    AGGREGATED = "aggregated"
    SCRAPED = "scraped"
    DERIVED = "derived"
    UNKNOWN = "unknown"

class AnomalyType(Enum):
    PRICE_SPIKE = "price_spike"
    MISSING_FIELD = "missing_field"
    NEGATIVE_VALUE = "negative_value"
    STALE_DATA = "stale_data"
    CROSS_SOURCE_MISMATCH = "cross_source_mismatch"
    UNIT_AMBIGUITY = "unit_ambiguity"
    CORPORATE_ACTION = "corporate_action"
    OUTLIER = "outlier"

@dataclass
class FreshnessScore:
    score: float
    data_timestamp: str
    query_timestamp: str
    max_delay_seconds: int
    is_stale: bool = False

    @classmethod
    def compute(
        cls,
        data_time: datetime,
        max_delay_seconds: int,
        query_time: Optional[datetime] = None,
    ) -> "FreshnessScore":
        if query_time is None:
            query_time = datetime.now(timezone.utc)
        age = (query_time - data_time).total_seconds()
        score = max(0.0, 1.0 - age / max_delay_seconds) if max_delay_seconds > 0 else 1.0
        return cls(
            score=round(score, 4),
            data_timestamp=data_time.isoformat(),
            query_timestamp=query_time.isoformat(),
            max_delay_seconds=max_delay_seconds,
            is_stale=score <= 0.0,
        )

@dataclass
class AnomalyFlag:
    anomaly_type: str
    field_name: str
    value: Any
    description: str
    severity: str = "warn"
    suggested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ETLRecord:
    record_id: str = ""
    raw_value: Any = None
    raw_unit: str = ""
    raw_currency: str = ""
    source: str = ""
    source_field: str = ""
    source_quality: str = "unknown"

    normalized_value: Optional[float] = None
    normalized_unit: str = ""
    normalized_currency: str = ""
    fx_rate_used: Optional[float] = None
    fx_rate_date: str = ""
    fx_rate_source: str = ""

    entity: str = ""
    field_name: str = ""
    period: str = ""
    data_timestamp: str = ""
    query_timestamp: str = ""
    freshness: Optional[FreshnessScore] = None

    anomalies: list[AnomalyFlag] = field(default_factory=list)
    is_validated: bool = False
    validation_notes: str = ""

    def __post_init__(self) -> None:
        if not self.record_id:
            self.record_id = f"etl_{uuid.uuid4().hex[:10]}"
        if not self.query_timestamp:
            self.query_timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0

    @property
    def has_errors(self) -> bool:
        return any(a.severity == "error" for a in self.anomalies)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def citation(self) -> str:
        val = self.normalized_value if self.normalized_value is not None else self.raw_value
        currency_prefix = f"{self.normalized_currency} " if self.normalized_currency else ""
        return (
            f"{currency_prefix}{val} "
            f"(来源: {self.source} | 时间: {self.data_timestamp} | 字段: {self.source_field})"
        )

_CN_UNITS: dict[str, float] = {
    "亿": 1e8,
    "万亿": 1e12,
    "万": 1e4,
    "千": 1e3,
    "百": 1e2,
}

_EN_UNITS: dict[str, float] = {
    "T": 1e12, "t": 1e12,
    "B": 1e9,  "b": 1e9,
    "M": 1e6,  "m": 1e6,
    "K": 1e3,  "k": 1e3,
    "billion": 1e9, "million": 1e6, "thousand": 1e3,
    "trillion": 1e12,
}

class UnitNormalizer:

    @staticmethod
    def normalize(value: Any, unit: str = "") -> tuple[Optional[float], str, list[AnomalyFlag]]:
        anomalies: list[AnomalyFlag] = []

        if value is None:
            anomalies.append(AnomalyFlag(
                anomaly_type=AnomalyType.MISSING_FIELD.value,
                field_name="value",
                value=None,
                description="Value is None",
                severity="error",
                suggested_action="discard",
            ))
            return None, "", anomalies

        if isinstance(value, (int, float)) and not unit:
            return float(value), "absolute", anomalies

        text = str(value).strip().replace(",", "")

        for cn_unit, multiplier in _CN_UNITS.items():
            if cn_unit in text or cn_unit in unit:
                num_str = re.sub(r"[^\d.\-]", "", text.replace(cn_unit, ""))
                try:
                    return float(num_str) * multiplier, cn_unit, anomalies
                except ValueError:
                    anomalies.append(AnomalyFlag(
                        anomaly_type=AnomalyType.UNIT_AMBIGUITY.value,
                        field_name="value",
                        value=text,
                        description=f"Cannot parse numeric from '{text}' with unit '{cn_unit}'",
                        severity="error",
                    ))
                    return None, cn_unit, anomalies

        for en_unit, multiplier in _EN_UNITS.items():
            if en_unit in text or en_unit.lower() in unit.lower():
                num_str = re.sub(r"[^\d.\-]", "", text)
                try:
                    return float(num_str) * multiplier, en_unit, anomalies
                except ValueError:
                    anomalies.append(AnomalyFlag(
                        anomaly_type=AnomalyType.UNIT_AMBIGUITY.value,
                        field_name="value",
                        value=text,
                        description=f"Cannot parse numeric from '{text}' with unit '{en_unit}'",
                        severity="error",
                    ))
                    return None, en_unit, anomalies

        num_str = re.sub(r"[^\d.\-]", "", text)
        if num_str:
            try:
                return float(num_str), unit or "absolute", anomalies
            except ValueError:
                pass

        anomalies.append(AnomalyFlag(
            anomaly_type=AnomalyType.UNIT_AMBIGUITY.value,
            field_name="value",
            value=text,
            description=f"Cannot parse: '{text}'",
            severity="error",
        ))
        return None, "", anomalies

_CURRENCY_SYMBOLS: dict[str, str] = {
    "¥": "CNY", "￥": "CNY", "$": "USD", "€": "EUR",
    "£": "GBP", "S$": "SGD", "HK$": "HKD",
}

class CurrencyNormalizer:

    def __init__(self, target_currency: str = "CNY") -> None:
        self.target_currency = target_currency.upper()

    def normalize(
        self,
        value: float,
        source_currency: str,
        fx_rate: Optional[float] = None,
        fx_date: str = "",
        fx_source: str = "",
    ) -> tuple[float, dict[str, Any]]:
        src = self._resolve_currency(source_currency)
        tgt = self.target_currency

        if src == tgt:
            return value, {
                "fx_rate": 1.0,
                "fx_date": fx_date,
                "fx_source": "identity",
                "source_currency": src,
                "target_currency": tgt,
            }

        if fx_rate is not None:
            rate = fx_rate
            source_label = fx_source or "user_provided"
        else:
            raise ValueError(f"No FX rate provided for {src}→{tgt}")

        converted = value * rate
        return converted, {
            "fx_rate": round(rate, 6),
            "fx_date": fx_date,
            "fx_source": source_label,
            "source_currency": src,
            "target_currency": tgt,
        }

    @staticmethod
    def _resolve_currency(raw: str) -> str:
        raw = raw.strip()
        if raw in _CURRENCY_SYMBOLS:
            return _CURRENCY_SYMBOLS[raw]
        return raw.upper()

class AnomalyDetector:

    def __init__(
        self,
        price_spike_threshold: float = 0.20,
        staleness_seconds: int = 86400,
    ) -> None:
        self.price_spike_threshold = price_spike_threshold
        self.staleness_seconds = staleness_seconds

    def check(
        self,
        record: ETLRecord,
        prior_value: Optional[float] = None,
    ) -> list[AnomalyFlag]:
        flags: list[AnomalyFlag] = []

        if record.normalized_value is None:
            flags.append(AnomalyFlag(
                anomaly_type=AnomalyType.MISSING_FIELD.value,
                field_name=record.field_name,
                value=record.raw_value,
                description=f"Normalized value is None for {record.field_name}",
                severity="error",
                suggested_action="verify",
            ))

        positive_fields = {"price", "volume", "market_cap", "revenue", "pe_ratio"}
        if (
            record.normalized_value is not None
            and record.normalized_value < 0
            and record.field_name.lower() in positive_fields
        ):
            flags.append(AnomalyFlag(
                anomaly_type=AnomalyType.NEGATIVE_VALUE.value,
                field_name=record.field_name,
                value=record.normalized_value,
                description=f"{record.field_name} should not be negative: {record.normalized_value}",
                severity="error",
                suggested_action="discard",
            ))

        if (
            prior_value is not None
            and record.normalized_value is not None
            and prior_value != 0
        ):
            change = abs(record.normalized_value - prior_value) / abs(prior_value)
            if change > self.price_spike_threshold:
                flags.append(AnomalyFlag(
                    anomaly_type=AnomalyType.PRICE_SPIKE.value,
                    field_name=record.field_name,
                    value=record.normalized_value,
                    description=(
                        f"Change of {change:.1%} from prior value {prior_value}. "
                        f"Possible corporate action (split/dividend/rights issue)."
                    ),
                    severity="warn",
                    suggested_action="verify",
                ))

        if record.freshness and record.freshness.is_stale:
            flags.append(AnomalyFlag(
                anomaly_type=AnomalyType.STALE_DATA.value,
                field_name=record.field_name,
                value=record.data_timestamp,
                description=(
                    f"Data freshness score {record.freshness.score:.2f} "
                    f"(threshold: {record.freshness.max_delay_seconds}s)"
                ),
                severity="warn",
                suggested_action="flag_for_review",
            ))

        return flags

    def check_cross_source(
        self,
        records: list[ETLRecord],
        tolerance: float = 0.02,
    ) -> list[AnomalyFlag]:
        flags: list[AnomalyFlag] = []
        values = [
            (r.source, r.normalized_value)
            for r in records
            if r.normalized_value is not None
        ]

        if len(values) < 2:
            return flags

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                src_a, val_a = values[i]
                src_b, val_b = values[j]
                if val_a == 0 and val_b == 0:
                    continue
                denominator = max(abs(val_a), abs(val_b), 1e-9)
                diff = abs(val_a - val_b) / denominator
                if diff > tolerance:
                    flags.append(AnomalyFlag(
                        anomaly_type=AnomalyType.CROSS_SOURCE_MISMATCH.value,
                        field_name=records[0].field_name,
                        value={"source_a": src_a, "value_a": val_a, "source_b": src_b, "value_b": val_b},
                        description=(
                            f"Cross-source mismatch: {src_a}={val_a} vs {src_b}={val_b} "
                            f"(diff={diff:.2%}, tolerance={tolerance:.2%})"
                        ),
                        severity="warn",
                        suggested_action="verify",
                    ))

        return flags

class DataValidator:

    def __init__(self) -> None:
        self._unit_normalizer = UnitNormalizer()
        self._anomaly_detector = AnomalyDetector()

    def validate_record(
        self,
        record: ETLRecord,
        prior_value: Optional[float] = None,
    ) -> ETLRecord:
        if record.raw_value is not None and record.normalized_value is None:
            norm_val, unit_used, unit_anomalies = self._unit_normalizer.normalize(
                record.raw_value, record.raw_unit
            )
            record.normalized_value = norm_val
            record.normalized_unit = unit_used
            record.anomalies.extend(unit_anomalies)

        anomalies = self._anomaly_detector.check(record, prior_value)
        record.anomalies.extend(anomalies)

        record.is_validated = True
        if record.has_errors:
            record.validation_notes = "⚠️ 验证失败，需要人工复核"
        elif record.has_anomalies:
            record.validation_notes = "⚠️ 检测到异常，建议交叉验证"
        else:
            record.validation_notes = "✅ 验证通过"

        return record

    def validate_batch(
        self,
        records: list[ETLRecord],
    ) -> tuple[list[ETLRecord], dict[str, Any]]:
        validated = [self.validate_record(r) for r in records]

        groups: dict[str, list[ETLRecord]] = {}
        for r in validated:
            key = f"{r.entity}:{r.field_name}"
            groups.setdefault(key, []).append(r)

        cross_source_flags: list[AnomalyFlag] = []
        for key, group in groups.items():
            if len(group) >= 2:
                flags = self._anomaly_detector.check_cross_source(group)
                cross_source_flags.extend(flags)
                for r in group:
                    r.anomalies.extend(flags)

        summary = {
            "total_records": len(validated),
            "passed": sum(1 for r in validated if not r.has_anomalies),
            "warnings": sum(1 for r in validated if r.has_anomalies and not r.has_errors),
            "errors": sum(1 for r in validated if r.has_errors),
            "cross_source_issues": len(cross_source_flags),
            "pass_rate": (
                sum(1 for r in validated if not r.has_anomalies) / len(validated)
                if validated else 1.0
            ),
        }

        return validated, summary
