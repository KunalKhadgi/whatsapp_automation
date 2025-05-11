from typing import Dict
from chromadb.api import API
from chromadb.api.models.Collection import Collection
from tqdm import tqdm
from more_itertools import chunked

from chroma_migrate.utils import migrate_embedding_metadata, validate_collection_metadata

CHUNK_SIZE = 1000

def migrate_from_remote_chroma(from_api: API, to_api: API):
    
    print("Loading Existing Collections...")
    from_collections = from_api.list_collections()

    if len(from_collections) == 0:
        print("No collections found, exiting...")
        return False

    print("Validating collection metadata...")
    from_collection_to_metadata = {}
    for collection in from_collections:
        from_collection_to_metadata[collection.name] = validate_collection_metadata(collection.metadata)

    print("Migrating existing collections...")
    from_collection_to_to_collection: Dict[str, Collection] = {}
    total_embeddings = 0
    for from_collection in from_collections:
        to_collection = to_api.get_or_create_collection(from_collection.name, from_collection_to_metadata[from_collection.name])
        total_embeddings += from_collection.count()
        from_collection_to_to_collection[from_collection.name] = to_collection
    
    print("Migrating existing embeddings...")
    with tqdm(total=total_embeddings) as pbar:
        for from_collection in from_collections:
            to_collection = from_collection_to_to_collection[from_collection.name]
            data = from_collection.get(include=["documents", "metadatas", "embeddings"])
            chunk_size = min(CHUNK_SIZE, len(data["ids"]))
            absolute_position = 0
            for chunk in chunked(data["ids"], chunk_size):
                ids = chunk
                embeddings = data["embeddings"][absolute_position:absolute_position+chunk_size]
                metadatas = data["metadatas"][absolute_position:absolute_position+chunk_size]
                for i, metadata in enumerate(metadatas):
                    metadatas[i] = migrate_embedding_metadata(metadata)

                documents = data["documents"][absolute_position:absolute_position+chunk_size]
                to_collection.add(ids, embeddings, metadatas, documents)
                pbar.update(len(chunk))
                absolute_position += len(chunk)
    
    print(f"Migrated {len(from_collections)} collections and {total_embeddings} embeddings")
    return True