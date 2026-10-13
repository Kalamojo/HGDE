import torch
from imagebind.models.imagebind_model import ImageBindModel
from imagebind import data
from imagebind.models.imagebind_model import ModalityType
import lancedb
import polars as pl
import random


def main(manifest_path: str, device: str, model: ImageBindModel, db_path: str, batch_size: int = 50, table_name: str = "pokemon"):
    df = pl.read_ndjson(manifest_path)
    
    use_image_flag = [True if i % 2 == 0 else False for i in range(df.shape[0])]
    use_image_flag = random.sample(use_image_flag, len(use_image_flag))
    df = df.with_columns(
        use_image=pl.Series(use_image_flag)
    )

    print("Df shape:", df.shape)

    image_df = df.filter(pl.col("use_image") == True)
    text_df = df.filter(pl.col("use_image") == False)

    images = image_df["image_path"]
    text = text_df["text"]

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

        # vision set
        image_df_part = image_df.slice(i, i+batch_size).to_dict(as_series=False)
        vals = list(image_df_part.values()) + [all_embeddings[round]['vision'].tolist()]
        keys = list(image_df_part.keys()) + ["vector"]

        image_ingest = [dict(zip(keys, values)) for values in zip(*vals)]

        # text set
        text_df_part = text_df.slice(i, i+batch_size).to_dict(as_series=False)
        vals = list(text_df_part.values()) + [all_embeddings[round]['text'].tolist()]
        keys = list(text_df_part.keys()) + ["vector"]

        text_ingest = [dict(zip(keys, values)) for values in zip(*vals)]

        to_ingest += image_ingest + text_ingest
        round += 1

    print(len(to_ingest))
    print(to_ingest[0])


    db = lancedb.connect(db_path)
    table = db.create_table(table_name, data=to_ingest, mode="overwrite")
    print(table.search().to_polars())
    

if __name__ == "__main__":
    manifest_path = "data/pokemon_data/manifest.jsonl"
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

    main(manifest_path, device, model, db_path)
