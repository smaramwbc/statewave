"""SQLAlchemy async engine and session factory."""

from typing import AsyncGenerator

from sqlalchemy import event, exc
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from server.core.config import settings

# Lazy engine initialization to avoid event loop binding at import time
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _install_outage_normalization(engine: AsyncEngine) -> None:
    """Surface database outages uniformly as ``OperationalError``.

    The asyncpg dialect leaks two outage shapes that bypass SQLAlchemy's usual
    error translation, so the app-level ``OperationalError`` handler
    (``server/core/errors.py``) would never see them:

    - connect-time network failures (DB down/unreachable/DNS) escape as raw
      ``OSError`` subclasses (``ConnectionRefusedError``, ``socket.gaierror``);
    - a connection dying mid-query surfaces as a bare ``DBAPIError`` with
      ``connection_invalidated=True`` instead of ``OperationalError``.

    Both are re-wrapped here so callers see one exception type for "the
    database is temporarily unavailable". Auth/config mistakes (bad password,
    unknown database) and genuine SQL bugs take other exception types and are
    deliberately left alone.
    """

    @event.listens_for(engine.sync_engine, "do_connect")
    def _wrap_connect_oserror(dialect, conn_rec, cargs, cparams):
        try:
            return dialect.connect(*cargs, **cparams)
        except OSError as exc_:  # ConnectionRefusedError, gaierror, timeout…
            raise exc.OperationalError(
                "database connection failed", None, exc_
            ) from exc_

    @event.listens_for(engine.sync_engine, "handle_error")
    def _wrap_disconnect(context):
        if context.is_disconnect and not isinstance(
            context.sqlalchemy_exception, exc.OperationalError
        ):
            return exc.OperationalError(
                "database connection lost", None, context.original_exception
            )


def get_engine() -> AsyncEngine:
    """Get or create the async engine (lazy initialization)."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=300,
        )
        _install_outage_normalization(_engine)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory (lazy initialization)."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields a database session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose of the engine and reset state (for testing)."""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None


def set_engine_for_testing(
    engine: AsyncEngine | None, factory: async_sessionmaker[AsyncSession] | None
) -> tuple[AsyncEngine | None, async_sessionmaker[AsyncSession] | None]:
    """Override engine and factory for testing. Returns previous values for restoration."""
    global _engine, _async_session_factory
    prev_engine, prev_factory = _engine, _async_session_factory
    _engine = engine
    _async_session_factory = factory
    return prev_engine, prev_factory
