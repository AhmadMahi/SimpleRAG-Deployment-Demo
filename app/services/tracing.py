from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow


logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment: str) -> None:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
    except Exception as exc:  # pragma: no cover - backend specific
        logger.warning("MLflow setup failed: %s", exc)


@contextmanager
def traced_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    started = False
    start_time = time.perf_counter()

    try:
        nested = mlflow.active_run() is not None
        mlflow.start_run(run_name=run_name, nested=nested)
        started = True
        if tags:
            mlflow.set_tags(tags)
    except Exception as exc:  # pragma: no cover - backend specific
        logger.debug("MLflow run could not start: %s", exc)

    try:
        yield
    finally:
        if started:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            try:
                mlflow.log_metric("latency_ms", elapsed_ms)
            except Exception:
                pass
            try:
                mlflow.end_run()
            except Exception:
                pass


def log_params(params: dict[str, Any]) -> None:
    try:
        mlflow.log_params(params)
    except Exception:
        pass


def log_metric(name: str, value: float) -> None:
    try:
        mlflow.log_metric(name, value)
    except Exception:
        pass


def log_dict(payload: dict[str, Any], artifact_path: str) -> None:
    try:
        mlflow.log_dict(payload, artifact_path)
    except Exception:
        pass
