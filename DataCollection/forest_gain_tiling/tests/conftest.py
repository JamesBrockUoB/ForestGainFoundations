from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--bench-refresh",
        action="store_true",
        default=False,
        help="Force re-running EE benchmarks instead of using cached results",
    )


@pytest.fixture(scope="session")
def bench_refresh(request) -> bool:
    return bool(request.config.getoption("--bench-refresh"))


@pytest.fixture(scope="session")
def bench_cache_path() -> Path:
    return Path(__file__).resolve().parent / "benchmark_results" / "bench_cache.json"
