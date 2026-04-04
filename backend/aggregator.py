from dataclasses import dataclass
from typing import Any


BASE_WEIGHTS = {
    "classifier": 0.50,
    "source":     0.30,
    "linguistic": 0.20,
}


@dataclass
class ModuleSignal:
    name: str
    key: str
    real_prob: float
    has_data: bool
    weight: float
    reasoning: str


@dataclass
class AggregatedResult:
    prediction: str         # REAL / FAKE
    real_probability: float # 0.0 – 1.0
    fake_probability: float
    decisions: list[dict]


def _redistribute(signals: list[ModuleSignal]) -> None:
    missing_weight = sum(s.weight for s in signals if not s.has_data)
    if missing_weight == 0:
        return

    active = [s for s in signals if s.has_data]
    if not active:
        return

    total_active = sum(s.weight for s in active)
    for s in active:
        s.weight += missing_weight * (s.weight / total_active)

    for s in signals:
        if not s.has_data:
            s.weight = 0.0


def _classifier_signal(model_result: dict) -> ModuleSignal:
    real_prob = model_result["real_probability"]
    confidence = model_result["confidence"]
    prediction = model_result["prediction"]
    return ModuleSignal(
        name="DistilBERT Classifier",
        key="classifier",
        real_prob=real_prob,
        has_data=True,
        weight=BASE_WEIGHTS["classifier"],
        reasoning=f"Model predicts {prediction} with {confidence:.0%} confidence.",
    )


def _source_signal(source_result: dict) -> ModuleSignal:
    status = source_result.get("status", "NOT_FOUND")
    domain = source_result.get("domain", "")

    if status == "RELIABLE":
        return ModuleSignal(
            name="Source Verifier",
            key="source",
            real_prob=0.90,
            has_data=True,
            weight=BASE_WEIGHTS["source"],
            reasoning=f"{domain} is a known reliable source.",
        )
    elif status == "UNRELIABLE":
        return ModuleSignal(
            name="Source Verifier",
            key="source",
            real_prob=0.10,
            has_data=True,
            weight=BASE_WEIGHTS["source"],
            reasoning=f"{domain} is a known unreliable source.",
        )
    elif status == "MIXED":
        return ModuleSignal(
            name="Source Verifier",
            key="source",
            real_prob=0.50,
            has_data=True,
            weight=BASE_WEIGHTS["source"],
            reasoning=f"{domain} has mixed reliability rating.",
        )
    else:
        return ModuleSignal(
            name="Source Verifier",
            key="source",
            real_prob=0.50,
            has_data=False,
            weight=BASE_WEIGHTS["source"],
            reasoning=f"{domain} not found in reliability database — signal unavailable.",
        )


def _linguistic_signal(ling_result: Any) -> ModuleSignal:
    # linguistic score is 0.0 (fake) to 1.0 (real)
    score = ling_result.score
    flags = ling_result.flags
    flag_summary = f" Flags: {', '.join(f['code'] for f in flags[:3])}{'...' if len(flags) > 3 else ''}." if flags else ""
    return ModuleSignal(
        name="Linguistic Analyzer",
        key="linguistic",
        real_prob=score,
        has_data=True,
        weight=BASE_WEIGHTS["linguistic"],
        reasoning=f"Linguistic credibility score: {score:.0%}.{flag_summary}",
    )


def _apply_source_override(real_prob: float, source_result: dict) -> float:
    """
    If the domain is well known, increase the probability
    It prevents a biased model from fully overriding a reliable source
    """
    status = source_result.get("status")
    if status == "RELIABLE":
        # floor at 0.60 — cannot call a known-reliable source more than 40% fake
        return max(real_prob, 0.60)
    elif status == "UNRELIABLE":
        # ceil at 0.40 — cannot call a known-unreliable source more than 40% real
        return min(real_prob, 0.40)
    return real_prob


def aggregate(
    model_result: dict,
    source_result: dict,
    ling_result: Any,
) -> AggregatedResult:
    signals = [
        _classifier_signal(model_result),
        _source_signal(source_result),
        _linguistic_signal(ling_result),
    ]

    _redistribute(signals)

    # Weighted sum of real_prob across all active signals
    real_prob = sum(s.real_prob * s.weight for s in signals)

    # Source override clamp
    real_prob = _apply_source_override(real_prob, source_result)
    real_prob = round(max(0.0, min(1.0, real_prob)), 4)
    fake_prob = round(1.0 - real_prob, 4)

    prediction = "REAL" if real_prob >= 0.5 else "FAKE"

    decisions = [
        {
            "module": s.name,
            "real_probability": round(s.real_prob, 3),
            "effective_weight": round(s.weight, 3),
            "has_data": s.has_data,
            "reasoning": s.reasoning,
        }
        for s in signals
    ]

    return AggregatedResult(
        prediction=prediction,
        real_probability=real_prob,
        fake_probability=fake_prob,
        decisions=decisions,
    )
