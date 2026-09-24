"""Tests for the optional OpenTelemetry wrapper in ``server/core/tracing.py``.

The module's whole job is that the server behaves identically whether or not
the ``otel`` extra is installed, so both branches need coverage. CI installs
only the ``dev`` extra, which means the OTel-present branches never execute
here on their own, and asserting ``span(...) is None`` would only record that
opentelemetry happens to be missing (and would fail on a machine that has it).

So these tests drive both sides explicitly through the module's own seams:
``_HAS_OTEL`` / ``_tracer`` for :func:`span`, and substituted ``opentelemetry``
modules for the imports :func:`setup_tracing` performs lazily. That keeps every
test deterministic in either environment and makes a regression in the
OTel-present path fail in CI rather than in an operator's deployment.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from server.core import tracing


# ── test doubles ────────────────────────────────────────────────────────────


class _FakeSpan:
    """Stands in for an OTel span object handed back by the tracer."""


class _FakeSpanContext:
    """Context manager returned by ``start_as_current_span``."""

    def __init__(self, span: _FakeSpan) -> None:
        self.span = span
        self.entered = False
        self.exited = False
        self.exit_exc_type: type[BaseException] | None = None

    def __enter__(self) -> _FakeSpan:
        self.entered = True
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.exited = True
        self.exit_exc_type = exc_type
        return False  # never swallow


class _FakeTracer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.contexts: list[_FakeSpanContext] = []

    def start_as_current_span(self, name, attributes=None):
        self.calls.append((name, attributes))
        ctx = _FakeSpanContext(_FakeSpan())
        self.contexts.append(ctx)
        return ctx


class _FakeTraceApi:
    """Stands in for the module-level ``opentelemetry.trace`` API."""

    def __init__(self) -> None:
        self.providers: list[object] = []

    def set_tracer_provider(self, provider) -> None:
        self.providers.append(provider)


def _use_otel(monkeypatch, tracer: _FakeTracer | None) -> None:
    """Make the module believe the OTel API is installed."""
    monkeypatch.setattr(tracing, "_HAS_OTEL", True)
    monkeypatch.setattr(tracing, "_tracer", tracer)


def _use_no_otel(monkeypatch) -> None:
    """Make the module believe the OTel API is absent, whatever is installed."""
    monkeypatch.setattr(tracing, "_HAS_OTEL", False)
    monkeypatch.setattr(tracing, "_tracer", None)


# ── module import contract ──────────────────────────────────────────────────


def test_has_otel_flag_matches_whether_opentelemetry_is_importable():
    """The guard flag must reflect reality, not a hard-coded value.

    Every other assertion here overrides ``_HAS_OTEL`` on purpose, so this is
    the one test that pins the flag to the real environment: importing the
    module must never raise, and the flag must agree with what is installed.
    """
    installed = importlib.util.find_spec("opentelemetry") is not None
    assert tracing._HAS_OTEL is installed
    if not installed:
        assert tracing._tracer is None


# ── span(): OTel absent ─────────────────────────────────────────────────────


def test_span_without_otel_yields_none(monkeypatch):
    _use_no_otel(monkeypatch)
    with tracing.span("ingest-episode") as current:
        assert current is None


def test_span_without_otel_ignores_attributes(monkeypatch):
    """Callers pass attributes unconditionally; the no-op path must accept them."""
    _use_no_otel(monkeypatch)
    with tracing.span("search", attributes={"subject": "s1", "limit": 10}) as current:
        assert current is None


def test_span_without_otel_still_propagates_body_exceptions(monkeypatch):
    """A tracing wrapper must never swallow the error it was wrapping.

    ``@contextmanager`` generators silence exceptions if the yield is not
    wrapped correctly, which would turn a failing request into a silent
    success. Assert the error comes back out untouched.
    """
    _use_no_otel(monkeypatch)
    sentinel = RuntimeError("body failed")
    with pytest.raises(RuntimeError) as excinfo:
        with tracing.span("compile"):
            raise sentinel
    assert excinfo.value is sentinel


def test_span_falls_back_to_noop_when_tracer_is_missing(monkeypatch):
    """``_HAS_OTEL`` true but no tracer must not raise AttributeError."""
    _use_otel(monkeypatch, tracer=None)
    with tracing.span("get-context") as current:
        assert current is None


# ── span(): OTel present ────────────────────────────────────────────────────


def test_span_with_otel_opens_a_span_and_yields_it(monkeypatch):
    tracer = _FakeTracer()
    _use_otel(monkeypatch, tracer)

    with tracing.span("ingest-episode") as current:
        assert current is tracer.contexts[0].span
        assert tracer.contexts[0].entered is True
        assert tracer.contexts[0].exited is False

    assert tracer.calls == [("ingest-episode", {})]
    assert tracer.contexts[0].exited is True


def test_span_with_otel_forwards_attributes(monkeypatch):
    tracer = _FakeTracer()
    _use_otel(monkeypatch, tracer)
    attributes = {"subject": "user:42", "limit": 10}

    with tracing.span("search", attributes=attributes):
        pass

    assert tracer.calls == [("search", attributes)]


def test_span_with_otel_normalises_missing_attributes_to_a_dict(monkeypatch):
    """The OTel API rejects a bare ``None``, hence the ``or {}`` in the wrapper."""
    tracer = _FakeTracer()
    _use_otel(monkeypatch, tracer)

    with tracing.span("compile"):
        pass

    name, attributes = tracer.calls[0]
    assert attributes == {}
    assert attributes is not None


def test_span_with_otel_closes_the_span_when_the_body_raises(monkeypatch):
    """The span has to be ended, and told about the error, on the failure path."""
    tracer = _FakeTracer()
    _use_otel(monkeypatch, tracer)

    with pytest.raises(ValueError):
        with tracing.span("compile"):
            raise ValueError("boom")

    ctx = tracer.contexts[0]
    assert ctx.exited is True
    assert ctx.exit_exc_type is ValueError


def test_nested_spans_each_get_their_own_span(monkeypatch):
    tracer = _FakeTracer()
    _use_otel(monkeypatch, tracer)

    with tracing.span("outer") as outer:
        with tracing.span("inner") as inner:
            assert outer is not inner

    assert [name for name, _ in tracer.calls] == ["outer", "inner"]


# ── setup_tracing() ─────────────────────────────────────────────────────────


def test_setup_tracing_without_otel_configures_nothing(monkeypatch):
    """Not just "does not raise": it must not touch the global tracer provider."""
    trace_api = _FakeTraceApi()
    monkeypatch.setattr(tracing, "trace", trace_api, raising=False)
    _use_no_otel(monkeypatch)

    assert tracing.setup_tracing(service_name="statewave-test") is None
    assert trace_api.providers == []


def _install_fake_otel_sdk(monkeypatch, *, with_exporter: bool) -> dict:
    """Register stand-ins for the modules ``setup_tracing`` imports lazily.

    Returns a record of what each stand-in was constructed with, so the caller
    can assert on the wiring instead of only on the absence of an exception.
    """
    record: dict = {"resources": [], "providers": [], "exporters": [], "processors": []}

    class FakeResource:
        def __init__(self, attributes):
            self.attributes = attributes

        @classmethod
        def create(cls, attributes):
            resource = cls(attributes)
            record["resources"].append(resource)
            return resource

    class FakeTracerProvider:
        def __init__(self, resource=None):
            self.resource = resource
            self.span_processors: list[object] = []
            record["providers"].append(self)

        def add_span_processor(self, processor):
            self.span_processors.append(processor)

    class FakeBatchSpanProcessor:
        def __init__(self, exporter):
            self.exporter = exporter
            record["processors"].append(self)

    class FakeOTLPSpanExporter:
        def __init__(self):
            record["exporters"].append(self)

    modules = {
        "opentelemetry.sdk.resources": {"Resource": FakeResource},
        "opentelemetry.sdk.trace": {"TracerProvider": FakeTracerProvider},
        "opentelemetry.sdk.trace.export": {"BatchSpanProcessor": FakeBatchSpanProcessor},
    }
    if with_exporter:
        modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"] = {
            "OTLPSpanExporter": FakeOTLPSpanExporter
        }

    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    if not with_exporter:
        # A ``None`` entry makes the import machinery raise ImportError for
        # that exact name, so the "no exporter installed" branch is taken even
        # on a machine where the real OTLP exporter package is present.
        monkeypatch.setitem(
            sys.modules, "opentelemetry.exporter.otlp.proto.grpc.trace_exporter", None
        )

    return record


def test_setup_tracing_with_otel_registers_a_provider_named_for_the_service(monkeypatch):
    trace_api = _FakeTraceApi()
    monkeypatch.setattr(tracing, "trace", trace_api, raising=False)
    monkeypatch.setattr(tracing, "_HAS_OTEL", True)
    record = _install_fake_otel_sdk(monkeypatch, with_exporter=True)

    tracing.setup_tracing(service_name="statewave-test")

    assert [r.attributes for r in record["resources"]] == [{"service.name": "statewave-test"}]
    assert len(record["providers"]) == 1
    provider = record["providers"][0]
    assert provider.resource is record["resources"][0]
    assert trace_api.providers == [provider]


def test_setup_tracing_defaults_the_service_name_to_statewave(monkeypatch):
    """``server/app.py`` calls ``setup_tracing()`` with no arguments."""
    monkeypatch.setattr(tracing, "trace", _FakeTraceApi(), raising=False)
    monkeypatch.setattr(tracing, "_HAS_OTEL", True)
    record = _install_fake_otel_sdk(monkeypatch, with_exporter=True)

    tracing.setup_tracing()

    assert record["resources"][0].attributes == {"service.name": "statewave"}


def test_setup_tracing_installs_an_otlp_exporter_when_one_is_available(monkeypatch):
    monkeypatch.setattr(tracing, "trace", _FakeTraceApi(), raising=False)
    monkeypatch.setattr(tracing, "_HAS_OTEL", True)
    record = _install_fake_otel_sdk(monkeypatch, with_exporter=True)

    tracing.setup_tracing()

    provider = record["providers"][0]
    assert len(provider.span_processors) == 1
    processor = provider.span_processors[0]
    assert processor is record["processors"][0]
    assert processor.exporter is record["exporters"][0]


def test_setup_tracing_still_registers_the_provider_without_an_exporter(monkeypatch):
    """Operators bring their own exporter; a missing one is documented as fine.

    The provider must still be registered, so spans are produced and simply not
    exported, rather than the whole call failing on the import.
    """
    trace_api = _FakeTraceApi()
    monkeypatch.setattr(tracing, "trace", trace_api, raising=False)
    monkeypatch.setattr(tracing, "_HAS_OTEL", True)
    record = _install_fake_otel_sdk(monkeypatch, with_exporter=False)

    tracing.setup_tracing(service_name="statewave-test")

    assert len(record["providers"]) == 1
    provider = record["providers"][0]
    assert trace_api.providers == [provider]
    assert provider.span_processors == []
    assert record["processors"] == []
