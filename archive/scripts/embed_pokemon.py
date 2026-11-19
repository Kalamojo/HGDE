import torch
from imagebind.models.imagebind_model import ImageBindModel
from imagebind import data
from imagebind.models.imagebind_model import ModalityType
import lancedb
import polars as pl
import random


def main(manifest_path: str, metadata_path: str, device: str, model: ImageBindModel, db_path: str, batch_size: int = 50, table_name: str = "pokemon"):
    df = pl.read_ndjson(manifest_path)
    metadata_df = pl.read_ndjson(metadata_path)

    full_df = df.join(metadata_df.select(pl.exclude("name")), on="dex_num", how="left")

    print("Df shape:", full_df.shape)

    images = full_df["image_path"]
    text = full_df["text"]

    round = 0
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        print("Round", round)
        cur_input = {
            ModalityType.VISION: data.load_and_transform_vision_data(images[i:i+batch_size], device),
            ModalityType.TEXT: data.load_and_transform_text(text[i:i+batch_size], device),
        }
        print("input loaded")

        with torch.amp.autocast(device, dtype=torch.float16):
            with torch.no_grad():
                embeddings = model(cur_input)

        print(embeddings)
        print()
        all_embeddings.append(embeddings)
        round += 1


    round = 0
    to_ingest = []
    for i in range(0, len(images), batch_size):
        print("Round", round)

        # vision and text sets
        df_part = full_df.slice(i, i+batch_size).to_dict(as_series=False)
        vals = list(df_part.values()) + [all_embeddings[round]['vision'].tolist(), all_embeddings[round]['text'].tolist()]
        keys = list(df_part.keys()) + ["image_vector", "text_vector"]

        data_ingest = [dict(zip(keys, values)) for values in zip(*vals)]

        to_ingest += data_ingest
        round += 1

    print(len(to_ingest))
    print(to_ingest[0])


    db = lancedb.connect(db_path)
    table = db.create_table(table_name, data=to_ingest, mode="overwrite")
    print(table.search().to_polars())
    

if __name__ == "__main__":
    manifest_path = "data/pokemon_data/manifest.jsonl"
    metadata_path = "data/pokemon_data/metadata.jsonl"
    db_path = "data/pokemon_db"
    device = "cpu" # "cpu", "cuda:0"

    model = ImageBindModel(
            vision_embed_dim=1280,
            vision_num_blocks=32,
            vision_num_heads=16,
            text_embed_dim=1024,
            text_num_blocks=24,
            text_num_heads=16,
            out_embed_dim=1024,
            audio_drop_path=0.1,
            imu_drop_path=0.7,
        )
    print("Model load start")
    model.load_state_dict(torch.load("src/imagebind/.checkpoints/imagebind_huge.pth", weights_only=True))
    model.eval()
    model.to(device)
    print("Model done loading")

    main(manifest_path, metadata_path, device, model, db_path)
