"""SQLAlchemy async engine and session factory."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from server.core.config import settings

# Lazy engine initialization to avoid event loop binding at import time
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


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
