from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="myshell-ai/OpenVoiceV2", repo_type="model", local_dir="./checkpoints_v2"
)
print("Done.")
