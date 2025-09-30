from huggingface_hub import login, snapshot_download


# 不指定 local_dir：会落到默认 HF_HOME (~/.cache/huggingface/hub)
path = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir=None,                    # 关键：让它用默认缓存
    local_dir_use_symlinks=True,       # 默认 True，体积更省（软链到缓存）
    # 也可以仅拉必要文件，下载更快更省空间：
    # allow_patterns=["*.safetensors", "*.json", "tokenizer.*"],
)
print("实际落地路径：", path)
