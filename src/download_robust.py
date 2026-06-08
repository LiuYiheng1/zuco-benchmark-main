from huggingface_hub import snapshot_download
import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

max_retries = 3
retry_delay = 10  # seconds

for attempt in range(max_retries):
    try:
        print(f"尝试下载 (尝试 {attempt + 1}/{max_retries})...")
        
        # 下载整个数据集
        snapshot_download(
            repo_id="lemonLHC/Zuco2.0",
            repo_type="dataset",
            local_dir=r"D:/pycharmproject/zuco-benchmark-main/data/Zuco2.0",
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print("下载完成！")
        break
        
    except Exception as e:
        print(f"下载失败: {e}")
        
        if attempt < max_retries - 1:
            print(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            retry_delay *= 2  # 指数退避
        else:
            print("已达到最大重试次数，下载失败。")
            print("请尝试通过浏览器下载：https://hf-mirror.com/datasets/lemonLHC/Zuco2.0")