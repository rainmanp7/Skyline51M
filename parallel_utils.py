
# parallel_utils.py
import asyncio
import concurrent.futures
import multiprocessing
import os

async def parallel_dynamic_adaptation_async(adaptation_tasks):
    tasks = [adjust_dynamic_adaptation(task) for task in adaptation_tasks]
    return await asyncio.gather(*tasks)

def choose_num_workers_based_on_complexity(complexity_factor):
    max_cores = os.cpu_count() or 1
    return min(max(1, complexity_factor // 10), max_cores)