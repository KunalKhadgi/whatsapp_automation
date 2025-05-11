from typing import Dict
import clickhouse_connect
from chromadb.api import API
from chromadb.api.models.Collection import Collection
from tqdm import tqdm
import json

from chroma_migrate.utils import migrate_embedding_metadata, validate_collection_metadata


def migrate_from_clickhouse(api: API, host: str, port: int):
    conn = clickhouse_connect.get_client(
            host=host,
            port=port,
        )

    print("Loading existing collections...")
    # Read the collections from clickhouse
    collections = conn.query("SELECT uuid, name, metadata FROM collections").result_rows

    if len(collections) == 0:
        print("No collections found, exiting...")
        return False

    print("Validating collection metadata...")
    from_collection_to_metadata = {}
    for collection in collections:
        metadata = json.loads(collection[2])
        from_collection_to_metadata[collection[1]] = validate_collection_metadata(metadata)

    # Create the collections in chromadb
    print("Migrating existing collections...")
    collection_uuid_to_chroma_collection: Dict[str, Collection] = {}
    for collection in collections:
        uuid = collection[0]
        name = collection[1]
        metadata = from_collection_to_metadata[name]
        coll = api.get_or_create_collection(name, metadata)
        collection_uuid_to_chroma_collection[uuid] = coll

    # -------------------------------------
    
    # Add the embeddings to the collections
    print("Migrating existing embeddings...")
    with tqdm(total=conn.query("SELECT count(*) FROM embeddings").result_rows[0][0]) as pbar:
        with conn.query_row_block_stream('SELECT uuid, collection_uuid, id, embedding, document, metadata FROM embeddings') as stream:
            for block in stream:
                for record in block:
                    uuid = record[0]
                    collection_uuid = record[1]
                    id = record[2]
                    embedding = record[3]
                    document = record[4]
                    metadata = json.loads(record[5])
                    metadata = migrate_embedding_metadata(metadata)
                    collection = collection_uuid_to_chroma_collection[collection_uuid]
                    collection.add(id, embedding, metadata, document)
                    pbar.update(1)
    
    print(f"Migrated {len(collections)} collections and {pbar.n} embeddings")
    return True