import hydra
import os
from huggingface_hub import (
    HfApi, 
    login
)

@hydra.main(version_base=None, config_path="../src/config/tools", config_name="hfdataset_upload")
def main(cfg):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    input(f"uploading dataset to {cfg.repo_id} -> \n\t from {cfg.upload_path} \n\t go ?" )
    api.upload_large_folder(
        repo_id=cfg.repo_id,
        repo_type="dataset",
        folder_path=cfg.upload_path,
    )


if __name__ == "__main__":
    # https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#upload-a-large-folder
    login()
    main()
    # v=cvEJ5WFk2KE__#00-18-00_00-21-00_label_A.npy: 100%|██████████████████████████████| 830k/830k [00:04<00:00, 198kB/s]
    # [2025-06-23 19:37:38,233][root][INFO] - Is done: exiting main loop
    # [2025-06-23 19:37:43,515][root][INFO] - Upload is complete! :)