import os
from huggingface_hub import snapshot_download

# 设置镜像源（对当前库完全有效）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 可选：清除可能存在的代理设置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

snapshot_download(
    repo_id="lemonLHC/Zuco2.0",
    repo_type="dataset",
    allow_patterns=["task2 - TSR/Matlab files/*"],  # 核心参数：只下载这个文件夹
    local_dir="D:/pycharmproject/zuco-benchmark-main/data",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4  # 可调整并发数，以提高稳定性
)