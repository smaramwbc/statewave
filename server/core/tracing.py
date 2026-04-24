"""Optional OpenTelemetry tracing.

Provides a thin wrapper that no-ops when opentelemetry is not installed.
To enable: ``pip install statewave[otel]`` and configure an exporter
(e.g. OTEL_EXPORTER_OTLP_ENDPOINT).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

try:
    from opentelemetry import trace

    _tracer = trace.get_tracer("statewave")
    _HAS_OTEL = True
except ImportError:  # pragma: no cover
    _HAS_OTEL = False
    _tracer = None  # type: ignore[assignment]


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start a trace span. No-ops gracefully when OTel is absent."""
    if not _HAS_OTEL or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name, attributes=attributes or {}) as s:
        yield s


def setup_tracing(service_name: str = "statewave") -> None:
    """Initialise OTel tracing if the SDK is installed.

    This does NOT install an exporter — operators bring their own
    (OTLP, Jaeger, Zipkin, etc.) via standard OTel env vars.
    """
    if not _HAS_OTEL:
        return
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Auto-detect exporters via opentelemetry-exporter-* packages
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    except ImportError:
        pass  # No OTLP exporter installed — that's fine
