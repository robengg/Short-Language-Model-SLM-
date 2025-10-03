from huggingface_hub import snapshot_download
snapshot_download("distilgpt2", local_dir="~/models/distilgpt2", local_dir_use_symlinks=False)
snapshot_download("rinna/japanese-gpt2-small", local_dir="~/models/rinna-jgpt2-small", local_dir_use_symlinks=False)
