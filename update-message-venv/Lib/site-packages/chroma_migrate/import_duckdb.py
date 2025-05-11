import os
from typing import Dict
import duckdb
from chromadb.api import API
from chromadb.api.models.Collection import Collection
from tqdm import tqdm
import json

from chroma_migrate.utils import migrate_embedding_metadata, validate_collection_metadata

def migrate_from_duckdb(api: API, persist_directory: str):
    # Load all the collections from the parquet files
    if not os.path.exists(persist_directory):
        raise Exception(f"Persist directory {persist_directory} does not exist")

    conn = duckdb.connect(read_only=False)

    print("Loading Existing Collections...")
    # Load the collections into duckdb
    collections_parquet_path = os.path.join(persist_directory, "chroma-collections.parquet")
    conn.execute(
            "CREATE TABLE collections (uuid STRING, name STRING, metadata STRING);"
        )
    conn.execute(
        f"INSERT INTO collections SELECT * FROM read_parquet('{collections_parquet_path}');"
    )

    # Read the collections from duckdb
    collections = conn.execute("SELECT uuid, name, metadata FROM collections").fetchall()

    if len(collections) == 0:
        print("No collections found, exiting...")
        return False

    print("Validating collection metadata...")
    from_collection_to_metadata = {}
    for collection in collections:
        uuid, name, metadata = collection
        metadata = json.loads(metadata)
        from_collection_to_metadata[name] = validate_collection_metadata(metadata)

    # Create the collections in chromadb
    print("Migrating existing collections...")
    collection_uuid_to_chroma_collection: Dict[str, Collection] = {}
    for collection in collections:
        uuid, name, metadata = collection
        coll = api.get_or_create_collection(name, from_collection_to_metadata[name])
        collection_uuid_to_chroma_collection[uuid] = coll

    # -------------------------------------
    
    # Load the embeddings into duckdb
    print("Migrating existing embeddings...")
    embeddings_parquet_path = os.path.join(persist_directory, "chroma-embeddings.parquet")
    conn.execute(
        "CREATE TABLE embeddings (collection_uuid STRING, uuid STRING, embedding DOUBLE[], document STRING, id STRING, metadata STRING);"
    )
    conn.execute(
        f"INSERT INTO embeddings SELECT * FROM read_parquet('{embeddings_parquet_path}');"
    )

    # Read the embeddings from duckdb
    embeddings = conn.execute("SELECT uuid, collection_uuid, id, embedding, document, metadata FROM embeddings").fetch_df()

    # Add the embeddings to the collections
    for record in tqdm(embeddings.itertuples(index=False), total=embeddings.shape[0]):
        uuid, collection_uuid, id, embedding, document, metadata = record
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception as e:
                print(f"Failed to load metadata for embedding {id} in collection {collection_uuid}. Malformed JSON")
        else:
            metadata = None
        if not isinstance(document, str):
            document = None
        metadata = migrate_embedding_metadata(metadata)
        collection = collection_uuid_to_chroma_collection[collection_uuid]
        collection.add(id, embedding, metadata, document)

    print(f"Migrated {len(collections)} collections and {embeddings.shape[0]} embeddings")

    return True

