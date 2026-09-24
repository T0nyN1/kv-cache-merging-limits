import modal

app = modal.App("download-wikitext")
data_volume = modal.Volume.from_name("ot_kv_data")

image = modal.Image.debian_slim().pip_install("datasets")


@app.function(
    image=image,
    volumes={"/ot_kv_data": data_volume},
    timeout=3600,
)
def download():
    import os

    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"
    os.makedirs("/ot_kv_data/datasets", exist_ok=True)

    from datasets import load_dataset

    print("Start downloading Wikitext dataset...")

    load_dataset("wikitext", "wikitext-2-raw-v1")

    data_volume.commit()
    print("Wikitext dataset has been downloaded to /ot_kv_data/datasets ！")


@app.local_entrypoint()
def main():
    download.remote()
