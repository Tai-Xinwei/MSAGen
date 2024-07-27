# -*- coding: utf-8 -*-
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Optional

from ai4s.jobq import JobQ, WorkSpecification
from ai4s.jobq import batch_enqueue as ai4s_batch_enqueue
from ai4s.jobq import launch_workers as ai4s_launch_workers
from ai4s.jobq.auth import get_token_credential
from azure.monitor.opentelemetry import configure_azure_monitor
from rich.logging import RichHandler

logger = logging.getLogger("ai4s.jobq")


class SkipLogFilter(logging.Filter):
    def __init__(self, exclude_prefixes):
        assert exclude_prefixes and len(exclude_prefixes) > 0
        self.exclude_prefixes = exclude_prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return all(
            [not record.name.startswith(prefix) for prefix in self.exclude_prefixes]
        )


def setup_logging(
    skip_ai4s_jobq_trace: bool = False,
    skip_ai4s_task_trace: bool = True,
    **rich_handler_kwargs,
):
    log_handler = RichHandler(**rich_handler_kwargs)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s %(message)s",
        handlers=[log_handler],
        force=True,
    )

    app_insights_connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if app_insights_connection_string:
        configure_azure_monitor(connection_string=app_insights_connection_string)
        azure_handler = logging.getLogger().handlers[-1]
        logging.getLogger("ai4s.jobq").setLevel(logging.INFO)
        if skip_ai4s_task_trace:
            azure_handler.addFilter(SkipLogFilter(["task."]))
        print(
            f"[WARNING] Azure Monitor logging enabled: {app_insights_connection_string}",
            flush=True,
        )
    else:
        print(
            "[WARNING] Azure Monitor logging disabled."
            "Set APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.",
            flush=True,
        )

    if skip_ai4s_jobq_trace:
        log_handler.addFilter(SkipLogFilter(["ai4s.jobq"]))


async def enqueue(
    storage_account: str,
    queue: str,
    workspec: WorkSpecification,
    max_workers: Optional[int] = None,
    list_task_workers: Optional[int] = None,
    enqueue_workers: Optional[int] = None,
    **kwargs,
):
    cpus = cpu_count()
    max_workers = max_workers or cpus
    list_task_workers = list_task_workers or cpus
    enqueue_workers = enqueue_workers or cpus * 4

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers))
    async with JobQ.from_storage_queue(
        queue, storage_account=storage_account, credential=get_token_credential()
    ) as jobq:
        await ai4s_batch_enqueue(
            jobq,
            workspec,
            num_list_task_workers=list_task_workers,
            num_enqueue_workers=enqueue_workers,
            **kwargs,
        )


async def launch_workers(
    storage_account: str, queue: str, workspec: WorkSpecification, **kwargs
):
    async with JobQ.from_storage_queue(
        queue, storage_account=storage_account, credential=get_token_credential()
    ) as jobq:
        await ai4s_launch_workers(jobq, workspec, **kwargs)
