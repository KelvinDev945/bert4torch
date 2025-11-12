import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Callable
import socket


# ============ 分布式环境初始化 ============

def setup_distributed(
    backend: str = 'nccl',
    init_method: Optional[str] = None,
    timeout_minutes: int = 30,
) -> tuple[int, int, int]:
    """初始化分布式训练环境

    Args:
        backend: 分布式后端 ('nccl', 'gloo', 'mpi')
        init_method: 初始化方法（如 'env://', 'tcp://...'）
        timeout_minutes: 超时时间（分钟）

    Returns:
        rank: 当前进程 rank
        world_size: 总进程数
        local_rank: 本地 rank
    """
    # 从环境变量获取分布式参数
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size == 1:
        # 单卡模式，不初始化分布式
        return rank, world_size, local_rank

    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # 初始化进程组
    if not dist.is_initialized():
        if init_method is None:
            init_method = 'env://'

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.timedelta(minutes=timeout_minutes),
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: Optional[int] = None) -> bool:
    """判断是否为主进程"""
    if rank is None:
        rank = get_rank()
    return rank == 0


def is_distributed() -> bool:
    """判断是否处于分布式模式"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """获取当前进程 rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """获取本地 rank"""
    return int(os.environ.get('LOCAL_RANK', 0))


# ============ 分布式通信 ============

def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM, async_op: bool = False):
    """All-reduce 操作"""
    if is_distributed():
        return dist.all_reduce(tensor, op=op, async_op=async_op)
    return tensor


def all_gather(tensor: torch.Tensor, async_op: bool = False):
    """All-gather 操作"""
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    if async_op:
        return dist.all_gather(tensor_list, tensor, async_op=True), tensor_list
    else:
        dist.all_gather(tensor_list, tensor)
        return tensor_list


def reduce_scatter(output: torch.Tensor, input_list: list, async_op: bool = False):
    """Reduce-scatter 操作（更高效的梯度归约）"""
    if not is_distributed():
        return input_list[0] if input_list else output

    return dist.reduce_scatter(output, input_list, async_op=async_op)


def broadcast(tensor: torch.Tensor, src: int = 0, async_op: bool = False):
    """广播操作"""
    if is_distributed():
        return dist.broadcast(tensor, src=src, async_op=async_op)
    return tensor


def barrier():
    """同步所有进程"""
    if is_distributed():
        dist.barrier()


# ============ DDP 模型包装 ============

def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    output_device: Optional[int] = None,
    broadcast_buffers: bool = True,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
) -> torch.nn.Module:
    """将模型包装为 DDP

    Args:
        model: 要包装的模型
        device_ids: 设备 ID 列表
        output_device: 输出设备
        broadcast_buffers: 是否广播 buffer
        find_unused_parameters: 是否查找未使用的参数
        gradient_as_bucket_view: 梯度作为 bucket 视图（更高效）
        static_graph: 静态图模式（更快，但要求图结构不变）

    Returns:
        DDP 包装的模型
    """
    if not is_distributed():
        return model

    local_rank = get_local_rank()

    if device_ids is None and torch.cuda.is_available():
        device_ids = [local_rank]

    if output_device is None and torch.cuda.is_available():
        output_device = local_rank

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )

    return ddp_model


# ============ 异步梯度归约 ============

class AsyncGradientReducer:
    """异步梯度归约器

    在反向传播时异步归约梯度，与计算重叠以提高效率
    参考 modded-nanogpt 的实现
    """
    def __init__(
        self,
        model: torch.nn.Module,
        use_reduce_scatter: bool = False,
    ):
        self.model = model
        self.use_reduce_scatter = use_reduce_scatter
        self.handles = []
        self.enabled = is_distributed()
        self.world_size = get_world_size()

        if self.enabled:
            self._register_hooks()

    def _register_hooks(self):
        """注册反向传播钩子"""
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._grad_hook
                )

    def _grad_hook(self, param: torch.nn.Parameter):
        """梯度累积后的钩子"""
        if param.grad is None:
            return

        # 异步归约梯度
        if self.use_reduce_scatter:
            # Reduce-scatter: 更高效，每个 rank 只保留部分梯度
            # 这需要配合优化器的分片实现
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.AVG,
                async_op=True
            )
        else:
            # All-reduce: 所有 rank 都有完整梯度
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.AVG,
                async_op=True
            )

        self.handles.append(handle)

    def synchronize(self):
        """同步所有异步操作"""
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


# ============ 分布式数据采样 ============

def get_distributed_sampler(
    dataset,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
):
    """获取分布式数据采样器"""
    if not is_distributed():
        return None

    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


# ============ 分布式日志 ============

class DistributedLogger:
    """分布式日志记录器

    只在主进程打印日志
    """
    def __init__(self, rank: Optional[int] = None):
        self.rank = rank if rank is not None else get_rank()
        self.is_main = is_main_process(self.rank)

    def print(self, *args, **kwargs):
        """仅在主进程打印"""
        if self.is_main:
            print(*args, **kwargs)

    def log(self, msg: str, force: bool = False):
        """记录日志

        Args:
            msg: 日志消息
            force: 是否强制所有进程打印
        """
        if force:
            rank = self.rank
            print(f"[Rank {rank}] {msg}")
        elif self.is_main:
            print(msg)


# ============ 分布式工具函数 ============

def sync_across_processes(value: float) -> float:
    """在所有进程间同步数值（求平均）"""
    if not is_distributed():
        return value

    tensor = torch.tensor(value, device='cuda' if torch.cuda.is_available() else 'cpu')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / get_world_size()


def gather_object(obj, dst: int = 0):
    """收集所有进程的对象到指定进程"""
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    rank = get_rank()

    if rank == dst:
        obj_list = [None] * world_size
    else:
        obj_list = None

    dist.gather_object(obj, obj_list, dst=dst)

    return obj_list


def all_gather_object(obj):
    """收集所有进程的对象到所有进程"""
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    obj_list = [None] * world_size
    dist.all_gather_object(obj_list, obj)

    return obj_list


def save_checkpoint_distributed(
    state_dict: dict,
    filepath: str,
    is_best: bool = False,
):
    """分布式保存检查点（仅主进程保存）"""
    if is_main_process():
        torch.save(state_dict, filepath)

        if is_best:
            import shutil
            best_path = filepath.replace('.pt', '_best.pt')
            shutil.copyfile(filepath, best_path)

    # 等待主进程保存完成
    barrier()


def load_checkpoint_distributed(filepath: str):
    """分布式加载检查点"""
    # 所有进程都从同一文件加载
    map_location = f'cuda:{get_local_rank()}' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)

    # 同步所有进程
    barrier()

    return checkpoint


# ============ 分布式环境检测 ============

def detect_distributed_env():
    """检测分布式环境配置"""
    info = {
        'backend': os.environ.get('BACKEND', 'nccl'),
        'rank': int(os.environ.get('RANK', 0)),
        'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
        'world_size': int(os.environ.get('WORLD_SIZE', 1)),
        'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
        'master_port': os.environ.get('MASTER_PORT', '29500'),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'hostname': socket.gethostname(),
    }

    return info


def print_distributed_info():
    """打印分布式环境信息"""
    info = detect_distributed_env()

    if is_main_process():
        print("=" * 50)
        print("分布式训练环境信息:")
        print("-" * 50)
        for key, value in info.items():
            print(f"{key:20s}: {value}")
        print("=" * 50)


# ============ 分布式初始化装饰器 ============

def distributed_init(func: Callable):
    """分布式初始化装饰器

    自动初始化和清理分布式环境
    """
    def wrapper(*args, **kwargs):
        # 初始化分布式
        rank, world_size, local_rank = setup_distributed()

        try:
            # 执行函数
            result = func(*args, rank=rank, world_size=world_size, local_rank=local_rank, **kwargs)
        finally:
            # 清理分布式
            cleanup_distributed()

        return result

    return wrapper


# ============ 分布式上下文管理器 ============

class DistributedContext:
    """分布式训练上下文管理器"""
    def __init__(
        self,
        backend: str = 'nccl',
        init_method: Optional[str] = None,
        timeout_minutes: int = 30,
    ):
        self.backend = backend
        self.init_method = init_method
        self.timeout_minutes = timeout_minutes
        self.rank = None
        self.world_size = None
        self.local_rank = None

    def __enter__(self):
        self.rank, self.world_size, self.local_rank = setup_distributed(
            backend=self.backend,
            init_method=self.init_method,
            timeout_minutes=self.timeout_minutes,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()


# ============ 分布式优化器状态同步 ============

def sync_optimizer_state(optimizer):
    """同步优化器状态（用于恢复训练）"""
    if not is_distributed():
        return

    # 广播优化器状态
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None:
                continue

            # 同步 optimizer state
            state = optimizer.state.get(param, {})
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    broadcast(value, src=0)


def average_gradients(model):
    """平均所有进程的梯度（手动实现）"""
    if not is_distributed():
        return

    world_size = get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
