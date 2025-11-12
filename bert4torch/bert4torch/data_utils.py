import os
import threading
import queue
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, Optional, Callable, List
import mmap


# ============ 异步数据预加载器 ============

class AsyncDataLoader:
    """异步数据加载器

    在后台线程预加载数据，与训练重叠以减少 I/O 等待
    参考 modded-nanogpt 的实现
    """
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        queue_size: int = 2,
    ):
        self.dataloader = dataloader
        self.device = device
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.stop_event = threading.Event()

    def _load_loop(self):
        """后台加载循环"""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break

                # 移动数据到设备
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device, non_blocking=True) if isinstance(b, torch.Tensor) else b
                            for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device, non_blocking=True)

                # 放入队列
                self.queue.put(batch)
        except Exception as e:
            self.queue.put(e)

    def start(self):
        """启动异步加载"""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._load_loop, daemon=True)
        self.thread.start()

    def __iter__(self):
        """迭代器"""
        self.start()
        return self

    def __next__(self):
        """获取下一批数据"""
        batch = self.queue.get()

        if isinstance(batch, Exception):
            raise batch

        if batch is None:
            raise StopIteration

        return batch

    def stop(self):
        """停止加载"""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()


# ============ 内存映射数据加载 ============

class MemoryMappedDataset(Dataset):
    """内存映射数据集

    用于大规模数据的高效加载，不占用内存
    """
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        dtype: np.dtype = np.uint16,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.dtype = dtype

        # 打开内存映射文件
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.data = np.frombuffer(self.mmap, dtype=dtype)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.num_samples = len(self.data) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        data = torch.from_numpy(self.data[start:end].astype(np.int64))
        return data

    def __del__(self):
        """清理内存映射"""
        if hasattr(self, 'mmap'):
            self.mmap.close()


# ============ BOS 对齐数据加载 ============

class BOSAlignedDataLoader:
    """BOS (Beginning of Sequence) 对齐数据加载器

    确保每个批次从文档边界开始，避免跨文档的序列
    参考 modded-nanogpt 的实现
    """
    def __init__(
        self,
        data: np.ndarray,
        bos_indices: List[int],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        drop_last: bool = True,
    ):
        self.data = data
        self.bos_indices = bos_indices
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.drop_last = drop_last

        # 预计算可用的起始位置
        self.valid_starts = [
            idx for idx in bos_indices
            if idx + seq_len <= len(data)
        ]

        self.num_batches = len(self.valid_starts) // batch_size
        if not drop_last and len(self.valid_starts) % batch_size != 0:
            self.num_batches += 1

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # 打乱起始位置
        indices = np.random.permutation(self.valid_starts)

        for i in range(self.num_batches):
            batch_starts = indices[i * self.batch_size:(i + 1) * self.batch_size]

            if len(batch_starts) < self.batch_size and self.drop_last:
                continue

            # 加载批次数据
            batch_data = []
            for start in batch_starts:
                seq = self.data[start:start + self.seq_len]
                batch_data.append(seq)

            # 转换为张量
            batch_tensor = torch.from_numpy(np.stack(batch_data)).long()
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)

            yield batch_tensor


# ============ 变长序列数据加载 ============

class VariableLengthDataLoader:
    """变长序列数据加载器

    支持不同长度的序列，动态填充到批次内最大长度
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        max_seq_len: int,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shuffle = shuffle
        self.drop_last = drop_last

    def collate_fn(self, batch):
        """整理批次数据

        动态填充到批次内最大长度
        """
        # 获取批次内最大长度
        lengths = [len(x) for x in batch]
        max_len = min(max(lengths), self.max_seq_len)

        # 填充
        padded_batch = []
        attention_masks = []

        for seq in batch:
            seq = seq[:max_len]  # 截断
            seq_len = len(seq)

            # 填充
            if seq_len < max_len:
                padding = torch.full((max_len - seq_len,), self.pad_token_id, dtype=seq.dtype)
                seq = torch.cat([seq, padding])

            # 注意力掩码
            mask = torch.cat([
                torch.ones(min(seq_len, max_len), dtype=torch.bool),
                torch.zeros(max(0, max_len - seq_len), dtype=torch.bool)
            ])

            padded_batch.append(seq)
            attention_masks.append(mask)

        # 堆叠
        batch_tensor = torch.stack(padded_batch)
        mask_tensor = torch.stack(attention_masks)

        return batch_tensor, mask_tensor

    def __len__(self):
        num_samples = len(self.dataset)
        if self.drop_last:
            return num_samples // self.batch_size
        return (num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(self)):
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]

            if len(batch_indices) < self.batch_size and self.drop_last:
                continue

            batch = [self.dataset[idx] for idx in batch_indices]
            batch_tensor, mask_tensor = self.collate_fn(batch)

            # 移动到设备
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)
            mask_tensor = mask_tensor.to(self.device, non_blocking=True)

            yield batch_tensor, mask_tensor


# ============ 预加载迭代器 ============

class PrefetchIterator:
    """数据预取迭代器

    在主线程计算的同时，后台线程预加载下一批数据
    """
    def __init__(
        self,
        iterator: Iterator,
        device: torch.device,
        prefetch_count: int = 1,
    ):
        self.iterator = iterator
        self.device = device
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.thread = None
        self.stop_event = threading.Event()

    def _prefetch_loop(self):
        """预取循环"""
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break

                # 移动到设备
                if isinstance(item, torch.Tensor):
                    item = item.to(self.device, non_blocking=True)
                elif isinstance(item, (tuple, list)):
                    item = [x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                           for x in item]
                elif isinstance(item, dict):
                    item = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                           for k, v in item.items()}

                self.queue.put(item)

            # 发送结束信号
            self.queue.put(None)
        except Exception as e:
            self.queue.put(e)

    def __iter__(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        item = self.queue.get()

        if item is None:
            raise StopIteration

        if isinstance(item, Exception):
            raise item

        return item

    def __del__(self):
        self.stop_event.set()


# ============ 工具函数 ============

def create_async_dataloader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    **kwargs
) -> AsyncDataLoader:
    """创建异步数据加载器

    Args:
        dataset: 数据集
        batch_size: 批次大小
        device: 设备
        num_workers: 工作线程数
        pin_memory: 是否锁页内存
        prefetch_factor: 预取因子
        **kwargs: 其他 DataLoader 参数

    Returns:
        异步数据加载器
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        **kwargs
    )

    return AsyncDataLoader(
        dataloader=dataloader,
        device=device,
        queue_size=prefetch_factor,
    )


def load_text_data(
    file_path: str,
    tokenizer: Optional[Callable] = None,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """加载文本数据

    Args:
        file_path: 文件路径
        tokenizer: 分词器函数
        max_length: 最大长度

    Returns:
        token ID 数组
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if tokenizer is not None:
        tokens = tokenizer(text)
    else:
        # 简单的字符级分词
        tokens = [ord(c) for c in text]

    if max_length is not None:
        tokens = tokens[:max_length]

    return np.array(tokens, dtype=np.int64)


def find_bos_indices(
    data: np.ndarray,
    bos_token_id: int,
) -> List[int]:
    """查找所有 BOS (Beginning of Sequence) 位置

    Args:
        data: 数据数组
        bos_token_id: BOS token ID

    Returns:
        BOS 位置列表
    """
    indices = np.where(data == bos_token_id)[0].tolist()

    # 添加数据开始位置
    if 0 not in indices:
        indices.insert(0, 0)

    return indices


def batch_to_device(batch, device: torch.device, non_blocking: bool = True):
    """将批次数据移动到设备

    Args:
        batch: 批次数据（tensor, tuple, list, dict）
        device: 目标设备
        non_blocking: 是否异步传输

    Returns:
        移动后的批次数据
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    elif isinstance(batch, (tuple, list)):
        return [batch_to_device(x, device, non_blocking) for x in batch]
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device, non_blocking) for k, v in batch.items()}
    else:
        return batch
