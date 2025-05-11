from typing import Any, Dict, Union

def validate_collection_metadata(metadata: Dict[Any, Any]) -> Dict[str, Union[str, int, float]]:
    if metadata is None:
        return metadata
    if len(metadata) == 0:
        return None
    # The new version of chroma only supports flat metadata. So we validate
    # that the metadata is is a flat dictionary of strings to ints, floats, or strings.
    for key, value in metadata.items():
        if not isinstance(value, (int, float, str)):
            raise ValueError(f"Metadata value {value} is not an int, float, or string. The new version of chroma only supports flat metadata. Please flatten your metadata using collection.modify() and try again.")
    return metadata

def migrate_embedding_metadata(metadata):
    return metadata if isinstance(metadata, dict) and len(metadata) > 0 else None
