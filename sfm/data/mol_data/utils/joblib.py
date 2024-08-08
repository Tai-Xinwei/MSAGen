# -*- coding: utf-8 -*-
import contextlib
import os
import time
import uuid
from datetime import datetime, timedelta
from threading import Thread
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class JobProgressState:
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.state = None

    @property
    def _state(self):
        if self.state is None:
            self.state = np.memmap(
                f"/tmp/joblib_progress_state_{uuid.uuid4().hex[:8]}",
                dtype=np.int32,
                mode="w+",
                shape=(self.num_tasks, 2),
            )
        return self.state

    def advance(self, task_id: int, num_processed: int):
        self._state[task_id, 0] += num_processed

    def complete(self, task_id: int):
        self._state[task_id, 1] = 1

    def __call__(self) -> Tuple[int, int]:
        return tuple(np.sum(self._state, axis=0))

    def clean(self):
        if self.state is not None:
            os.remove(self.state.filename)
            self.state = None


class JoblibProgress:
    def __init__(
        self,
        num_tasks: int,
        update_interval_sec: float = 0.5,
        description_fn: Optional[Callable[[int, timedelta, int, int], str]] = None,
    ):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            "<",
            TimeRemainingColumn(),
        )

        self.num_tasks = num_tasks
        self.state = JobProgressState(self.num_tasks)
        self.task_id = self.progress.add_task("[Joblib parallel]", total=num_tasks)
        self.update_interval = 0.5 if update_interval_sec <= 0 else update_interval_sec
        self.description_fn = description_fn

    def advance(self, task_id: int, num_processed: int):
        self.state.advance(task_id, num_processed)

    def complete(self, task_id: int):
        self.state.complete(task_id)

    def start(self):
        self.start_time = datetime.now()
        self.progress.start()

        def _update_progress():
            while self.update() < self.num_tasks:
                time.sleep(self.update_interval)

        self._update_thread = Thread(target=_update_progress)
        self._update_thread.start()
        return self

    def stop(self):
        self._update_thread.join()
        self.update()
        self.progress.stop()
        self.state.clean()

    def update(self):
        total_processed, completed_tasks = self.state()
        duration = datetime.now() - self.start_time
        speed = total_processed // duration.seconds if duration.seconds > 0 else 0

        if self.description_fn:
            desc = self.description_fn(
                total_processed, duration, speed, completed_tasks
            )
        else:
            desc = f"[[green]{total_processed} items processed[/green], [yellow]{speed} items/sec[/yellow]]"

        self.progress.update(
            self.task_id,
            completed=completed_tasks,
            description=desc,
            refresh=(completed_tasks == self.num_tasks),
        )
        return completed_tasks


@contextlib.contextmanager
def joblib_progress(
    num_tasks: int,
    update_interval_sec: float = 0.5,
    description_fn: Optional[Callable[[int, timedelta, int, int], str]] = None,
):
    progress = JoblibProgress(num_tasks, update_interval_sec, description_fn)
    try:
        progress.start()
        yield progress.state
    finally:
        progress.stop()


def parallel(
    tasks: List[Any],
    processor: Callable[[int, Any, JobProgressState], Any],
    num_workers: Optional[int] = None,
    progress_description_fn: Optional[Callable[[int, timedelta, int, int], str]] = None,
    **kwargs: Any,
) -> Iterable[Any]:
    num_workers = num_workers or os.cpu_count()
    desc_fn = progress_description_fn
    with joblib_progress(len(tasks), description_fn=desc_fn) as progress:
        for result in Parallel(n_jobs=num_workers, return_as="generator", **kwargs)(
            delayed(processor)(i, task, progress) for i, task in enumerate(tasks)
        ):
            yield result
