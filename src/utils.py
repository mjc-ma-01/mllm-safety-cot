

import logging
from typing import Any

def split_datasets(dataset: Any, rank: int, world_size: int) -> Any:
    """
    Args:
        dataset (Any): 原始数据集，必须支持 `len()` 和 `select()` 方法。
        rank (int): 当前任务编号，0 <= rank < world_size。
        world_size (int): 总任务数量。
    """
    dataset_size = len(dataset)
    block_size = (dataset_size + world_size - 1) // world_size  # 向上取整

    start_index = rank * block_size
    end_index = min((rank + 1) * block_size, dataset_size)
    
    logging.info(f"Total dataset size: {dataset_size}")
    logging.info(f"Block size: {block_size}")
    logging.info(f"Rank {rank}: Selecting data from index {start_index} to {end_index} (Total: {end_index - start_index})")
    print(f"Total dataset size: {dataset_size}")
    print(f"Block size: {block_size}")
    print(f"Rank {rank}: Selecting data from index {start_index} to {end_index} (Total: {end_index - start_index})")
    return dataset.select(range(start_index, end_index))
