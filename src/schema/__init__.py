
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

class Modality(Enum):
    TEXT = "text"
    TIMESERIES = "timeseries"
    TABLE = "table"
    GRAPH = "graph"
    IMAGE = "image"
    AUDIO = "audio"
    EVENT = "event"

@dataclass
class TextData:
    content: str = ""
    language: str = "auto"
    content_type: str = "news"
    entities: list[dict[str, Any]] = field(default_factory=list)
    sentiment: str = ""
    sentiment_score: float = 0.0

@dataclass
class TimeseriesData:
    columns: list[str] = field(default_factory=list)

    values: list[list[Any]] = field(default_factory=list)

    frequency: str = "daily"
    unit: str = ""
    asset: str = ""
    asset_type: str = ""
    start_date: str = ""
    end_date: str = ""
    adjustment: str = "none"

    @property
    def length(self) -> int:
        return len(self.values)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.values]

@dataclass
class TableData:
    headers: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    table_type: str = "generic"
    caption: str = ""
    column_units: list[str] = field(default_factory=list)
    period: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.headers))

@dataclass
class GraphData:
    nodes: list[dict[str, Any]] = field(default_factory=list)

    edges: list[dict[str, Any]] = field(default_factory=list)

    graph_type: str = "directed"
    domain: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

@dataclass
class ImageData:
    path: str = ""
    base64: str = ""
    format: str = "png"
    image_type: str = "chart"
    caption: str = ""
    width: int = 0
    height: int = 0
    page_number: int = 0

@dataclass
class AudioData:
    path: str = ""
    format: str = "wav"
    audio_type: str = "earnings_call"
    duration_seconds: float = 0.0
    transcript: str = ""
    language: str = "auto"
    speaker_labels: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class EventData:
    event_type: str = ""
    event_date: str = ""
    description: str = ""
    impact: str = ""
    magnitude: str = ""
    affected_entities: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

_MODALITY_CLASSES = {
    "text": TextData,
    "timeseries": TimeseriesData,
    "table": TableData,
    "graph": GraphData,
    "image": ImageData,
    "audio": AudioData,
    "event": EventData,
}

@dataclass
class ModalityBlock:
    modality: str
    data: Any = None
    source: str = ""
    source_url: str = ""
    source_quality: str = "aggregated"
    timestamp: str = ""
    data_timestamp: str = ""
    freshness_score: float = 1.0
    confidence: float = 1.0
    is_validated: bool = False
    validation_notes: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "modality": self.modality,
            "data": asdict(self.data) if self.data else {},
            "source": self.source,
            "source_url": self.source_url,
            "source_quality": self.source_quality,
            "timestamp": self.timestamp,
            "data_timestamp": self.data_timestamp,
            "freshness_score": self.freshness_score,
            "confidence": self.confidence,
            "is_validated": self.is_validated,
            "validation_notes": self.validation_notes,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModalityBlock":
        modality = d.get("modality", "text")
        data_dict = d.get("data", {})
        data_cls = _MODALITY_CLASSES.get(modality, TextData)
        data_obj = data_cls(**data_dict)
        return cls(
            modality=modality,
            data=data_obj,
            source=d.get("source", ""),
            source_url=d.get("source_url", ""),
            source_quality=d.get("source_quality", "aggregated"),
            timestamp=d.get("timestamp", ""),
            data_timestamp=d.get("data_timestamp", ""),
            freshness_score=d.get("freshness_score", 1.0),
            confidence=d.get("confidence", 1.0),
            is_validated=d.get("is_validated", False),
            validation_notes=d.get("validation_notes", ""),
        )

@dataclass
class FinanceClawSample:
    id: str = ""
    question: str = ""
    instruction: str = ""
    answer: Any = None
    answer_type: str = ""
    modalities: list[ModalityBlock] = field(default_factory=list)
    domain: str = ""
    topic: str = ""
    difficulty: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    data_cutoff: str = ""
    resolved_at: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"sample_{uuid.uuid4().hex[:10]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def get_modality(self, modality: str) -> list[ModalityBlock]:
        return [m for m in self.modalities if m.modality == modality]

    def get_text(self) -> list[TextData]:
        return [m.data for m in self.modalities if m.modality == "text"]

    def get_timeseries(self) -> list[TimeseriesData]:
        return [m.data for m in self.modalities if m.modality == "timeseries"]

    def get_tables(self) -> list[TableData]:
        return [m.data for m in self.modalities if m.modality == "table"]

    def get_graphs(self) -> list[GraphData]:
        return [m.data for m in self.modalities if m.modality == "graph"]

    def get_images(self) -> list[ImageData]:
        return [m.data for m in self.modalities if m.modality == "image"]

    def get_events(self) -> list[EventData]:
        return [m.data for m in self.modalities if m.modality == "event"]

    @property
    def modality_types(self) -> list[str]:
        return sorted(set(m.modality for m in self.modalities))

    @property
    def is_multimodal(self) -> bool:
        return len(self.modality_types) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "instruction": self.instruction,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "modalities": [m.to_dict() for m in self.modalities],
            "domain": self.domain,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "created_at": self.created_at,
            "data_cutoff": self.data_cutoff,
            "resolved_at": self.resolved_at,
            "entities": self.entities,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FinanceClawSample":
        modalities = [ModalityBlock.from_dict(m) for m in d.get("modalities", [])]
        return cls(
            id=d.get("id", ""),
            question=d.get("question", ""),
            instruction=d.get("instruction", ""),
            answer=d.get("answer"),
            answer_type=d.get("answer_type", ""),
            modalities=modalities,
            domain=d.get("domain", ""),
            topic=d.get("topic", ""),
            difficulty=d.get("difficulty", ""),
            tags=d.get("tags", []),
            created_at=d.get("created_at", ""),
            data_cutoff=d.get("data_cutoff", ""),
            resolved_at=d.get("resolved_at", ""),
            entities=d.get("entities", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FinanceClawSample":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_legacy_text(cls, d: dict[str, Any]) -> "FinanceClawSample":
        text_content = d.get("text", d.get("context", ""))
        question = d.get("question", "")
        entities = d.get("entities", [])

        modalities: list[ModalityBlock] = []
        if text_content:
            text_data = TextData(
                content=text_content,
                entities=entities,
                sentiment=d.get("label", ""),
            )
            modalities.append(ModalityBlock(
                modality="text",
                data=text_data,
                source=d.get("source", "legacy"),
            ))

        return cls(
            id=d.get("id", ""),
            question=question,
            answer=d.get("answer"),
            answer_type=_infer_answer_type(d.get("answer")),
            modalities=modalities,
            domain=d.get("domain", ""),
            topic=d.get("category", d.get("aspect", "")),
            difficulty=d.get("difficulty", ""),
            tags=[],
            entities=[{"name": e["text"], "type": e["type"]} for e in entities if isinstance(e, dict)],
        )

def load_dataset(path: str | Path) -> list[FinanceClawSample]:
    path = Path(path)
    samples: list[FinanceClawSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "modalities" in d:
                samples.append(FinanceClawSample.from_dict(d))
            else:
                samples.append(FinanceClawSample.from_legacy_text(d))
    return samples

def save_dataset(samples: list[FinanceClawSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_jsonl() + "\n")

def _infer_answer_type(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, bool):
        return "yes_no"
    if isinstance(answer, (int, float)):
        return "number"
    if isinstance(answer, dict):
        return "json"
    if isinstance(answer, str):
        s = answer.lower().strip()
        if s in ("yes", "no", "是", "否"):
            return "yes_no"
        return "text"
    return "text"
