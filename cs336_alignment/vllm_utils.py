"""
Small vLLM helpers for server lifecycle, completion requests, and NCCL weight sync.
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


"""单条生成结果"""
@dataclass
class VLLMCompletion:
    text: str              # 生成的文本（不含 prompt）
    token_ids: list[int]   # 生成文本对应的 token id（vLLM 返回 return_token_ids=True 时才有）
    finish_reason: str | None  # 结束原因：'stop'（遇到 stop token/串）、'length'（达到 max_tokens）、None


"""服务器句柄，这是一个封装 vLLM 服务器生命周期 + 通信的类。创建它不会立刻启动服务器，要显式调 .start()。"""
@dataclass
class VLLMServer:
    model_id: str                          # HuggingFace 模型名或本地路径，vLLM 用它加载模型
    host: str = "127.0.0.1"                # 监听地址
    port: int = 8000                       # 监听端口
    gpu: int = 1                           # 用哪块物理 GPU（写进 CUDA_VISIBLE_DEVICES）
    seed: int = 0                          # 采样随机种子
    load_format: str = "auto"              # 权重加载格式；'auto' 让 vLLM 自己判断
    logging_level: str = "ERROR"           # vLLM 日志级别
    gpu_memory_utilization: float = 0.9    # vLLM 预占 GPU 显存比例
    launch_server: bool = True             # 是否由本进程启动服务器；若连外部已有服务器可设 False
    startup_timeout: int = 600             # 等待服务器就绪的上限（秒）
    shutdown_timeout: int = 30             # 关闭时等待优雅退出的上限

    def __post_init__(self) -> None:
        # base_url 后续所有 HTTP 请求都用它
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None            # 保存子进程句柄，用于关闭
        self.weight_sync_group = None  # NCCL 通信组，初始化后才有值

    def start(self) -> None:
        if self.launch_server:
            kill_existing_vllm_server(self.port)    # 先清掉可能占端口的旧 vLLM，避免端口冲突
            self.process = start_server(        # 真正启动子进程
                model_id=self.model_id,
                host=self.host,
                port=self.port,
                gpu=self.gpu,
                seed=self.seed,
                load_format=self.load_format,
                logging_level=self.logging_level,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            atexit.register(self.stop) # 进程退出时自动关服务器，防止泄漏
        wait_for_server(self.base_url, self.process, self.startup_timeout)      # 阻塞直到 /health 可用

    def stop(self) -> None:
        stop_server(self.process, timeout=self.shutdown_timeout)

    def init_weight_sync(self, policy_device: str):
        # 建立训练进程 <-> vLLM 的 NCCL 通道；policy_device 指定训练侧用哪块 GPU（default="cuda:0"）
        self.weight_sync_group = init_weight_sync(self.base_url, policy_device)
        return self.weight_sync_group

    def sync_policy_weights(self, policy: torch.nn.Module) -> None:
        # 把当前 policy 参数推给 vLLM。必须先 init_weight_sync。
        if self.weight_sync_group is None:
            raise RuntimeError("Call init_weight_sync before sync_policy_weights.")
        sync_policy_weights(policy, self.base_url, self.weight_sync_group)

    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]:
         # 转发给模块级 generate_completions，用 self.base_url / self.model_id
        return generate_completions(
            vllm_base_url=self.base_url,
            model_id=self.model_id,
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )


""" 极简 HTTP JSON 客户端"""
def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    # 把 payload 序列化成 JSON，发 HTTP 请求，读回 body 再解析成 dict
    # 没有 body（如 /pause 可能返回空）时返回 {}
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body)


def kill_existing_vllm_server(port: int) -> None:
    pattern = f"vllm serve .* --port {port}"
    try:
        result = subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
        if result.returncode == 0:
            time.sleep(2)
            subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
    except FileNotFoundError:
        pass


def start_server(
    model_id: str,
    host: str,
    port: int,
    gpu: int,
    seed: int,
    load_format: str,
    logging_level: str,
    gpu_memory_utilization: float = 0.9,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)   # 关键：让 vLLM 只看见指定 GPU
    env["VLLM_SERVER_DEV_MODE"] = "1"        # 开启开发模式，暴露 /update_weights 等内部端点
    env["VLLM_LOGGING_LEVEL"] = logging_level
    command = [
        "vllm",
        "serve",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype", "bfloat16",                # 用 bf16，省显存且训练常用
        "--enable-prefix-caching",            # 相同 prompt 前缀复用 KV cache，加速
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        "1",        # 单卡推理
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),        # 允许 NCCL 权重更新
        "--load-format",
        load_format,
    ]
    logger.info("Starting vLLM server: %s", " ".join(command))
    # start_new_session=True：让子进程自成进程组，方便后面按组 kill
    return subprocess.Popen(command, env=env, start_new_session=True)


def wait_for_server(base_url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}.")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM server at {base_url}.")


def stop_server(process: subprocess.Popen | None, timeout: int = 30) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def generate_completions(
    vllm_base_url: str,
    model_id: str,
    prompts: list[str],
    sampling_params: dict,
    batch_size: int | None = None,
) -> list[VLLMCompletion]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    
    # sampling_params 需含: temperature, max_tokens, n, seed（可选 stop）
    prompt_batches = [prompts]
    if batch_size is not None:
        # 把大 prompt 列表切成小批，避免单次 HTTP 请求过大 / 超时
        prompt_batches = [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]

    completions = []
    for prompt_batch in prompt_batches:
        payload = {
            "model": model_id,
            "prompt": prompt_batch, # vLLM 支持一次传多个 prompt
            "temperature": sampling_params["temperature"],
            "max_tokens": sampling_params["max_tokens"],
            "n": sampling_params["n"],      # 每个 prompt 生成几条（GRPO 里 = group_size）
            "seed": sampling_params["seed"],
            "return_token_ids": True,       # 关键：返回 token id，供后续算 log-prob
        }
        if sampling_params.get("stop") is not None:
            payload["stop"] = sampling_params["stop"]
            payload["include_stop_str_in_output"] = sampling_params.get("include_stop_str_in_output", False)

        response = _http_json("POST", f"{vllm_base_url}/v1/completions", payload, timeout=3600)
        choices = sorted(response["choices"], key=lambda choice: choice["index"])       # 按 index 排序，保证与 prompt 对应
        completions.extend(
            VLLMCompletion(
                text=choice["text"],
                token_ids=choice.get("token_ids") or [],
                finish_reason=choice.get("finish_reason"),
            )
            for choice in choices
        )
    return completions


def init_weight_sync(vllm_base_url: str, policy_device: str):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    # 1. 问 vLLM：你那边有几个 rank？（单卡 TP=1 时为 1）
    inference_world_size = _http_json("GET", f"{vllm_base_url}/get_world_size", timeout=10)["world_size"]
    # 2. 总 world_size = 推理侧 + 训练侧 1
    world_size = inference_world_size + 1
    master_address = get_ip()
    master_port = get_open_port()
    init_info = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": 1,       # vLLM 侧 rank 从 1 开始，训练侧 rank=0
        "world_size": world_size,
    }
    
    # 3. 训练进程固定用 policy_device
    torch.cuda.set_device(torch.device(policy_device))

    # 4. 关键：两边必须“同时”初始化，否则 NCCL 建链会挂
    #    所以一边异步发 HTTP 让 vLLM 加入，一边本地初始化 trainer 侧
    with ThreadPoolExecutor(max_workers=1) as executor:
        init_future = executor.submit(
            _http_json,
            "POST",
            f"{vllm_base_url}/init_weight_transfer_engine",
            {"init_info": init_info},
            60,
        )
        weight_sync_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_future.result()

    return weight_sync_group


def sync_policy_weights(policy: torch.nn.Module, vllm_base_url: str, weight_sync_group) -> None:
    """Copy policy weights into vLLM and invalidate caches derived from old weights."""
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    # 1. 收集 policy 的所有参数名、dtype、shape，告诉 vLLM 要接收什么
    weights = list(policy.named_parameters())
    update_info = {
        "names": [name for name, _ in weights],
        "dtype_names": [str(tensor.dtype).split(".")[-1] for _, tensor in weights], # 'bfloat16' 等
        "shapes": [list(tensor.shape) for _, tensor in weights],
        "packed": True,     # 打包传输，减少通信次数
    }

    torch.cuda.set_device(next(policy.parameters()).device)     # 在 policy 所在 GPU 上建 NCCL 通信
    _http_json("POST", f"{vllm_base_url}/pause", timeout=60)    # 暂停推理，避免更新到一半被读
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 同样必须并发：vLLM 侧准备接收 + 训练侧发送，否则死锁
        update_future = executor.submit(
            _http_json,
            "POST",
            f"{vllm_base_url}/update_weights",
            {"update_info": update_info},
            300,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iter(weights),
            trainer_args=NCCLTrainerSendWeightsArgs(
                group=weight_sync_group,
                packed=True,
            ),
        )
        update_future.result()
    _http_json("POST", f"{vllm_base_url}/reset_prefix_cache", timeout=60)   # 权重变了，旧 KV cache 失效
    _http_json("POST", f"{vllm_base_url}/resume", timeout=60)   # 恢复推理
