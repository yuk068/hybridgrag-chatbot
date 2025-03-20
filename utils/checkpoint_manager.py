import os
import json
from typing import Dict, Any

class CheckpointManager:
    def __init__(self, path):
        self.path = path
        self.data: Dict[str, Any] = {
            "processed_files": [],
            "converted_chunks": [],
            "inserted_documents": [],  # Track inserted documents by doc_id
            "inserted_batches": [],  # Track batch indices for logging purposes
            "processed_graph_docs": [],  # New field to track processed graph docs by doc_id
        }
        if os.path.exists(path):
            self.load()

    def load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)

    def is_file_processed(self, file_path: str, file_hash: int) -> bool:
        """Check if a file has been processed by its hash."""
        return file_hash in self.data["processed_files"]

    def add_processed_file(self, file_path: str, file_hash: int):
        """Mark a file as processed by its hash."""
        if file_hash not in self.data["processed_files"]:
            self.data["processed_files"].append(file_hash)
            self.save()

    def add_converted_chunk(self, chunk_hash: int):
        """Mark a chunk as converted by its hash."""
        if chunk_hash not in self.data["converted_chunks"]:
            self.data["converted_chunks"].append(chunk_hash)
            self.save()

    def is_graph_doc_processed(self, doc_id: str) -> bool:
        """Check if a graph document has been processed by its doc_id."""
        return doc_id in self.data["processed_graph_docs"]

    def add_processed_graph_doc(self, doc_id: str):
        """Mark a graph document as processed by its doc_id."""
        if doc_id not in self.data["processed_graph_docs"]:
            self.data["processed_graph_docs"].append(doc_id)
            self.save()

    def is_document_inserted(self, doc_hash: int) -> bool:
        """Check if a document has been inserted into Neo4j by its doc_hash."""
        return doc_hash in self.data["inserted_documents"]

    def add_inserted_document(self, doc_id: str):
        """Mark a document as inserted into Neo4j by its doc_id."""
        if doc_id not in self.data["inserted_documents"]:
            self.data["inserted_documents"].append(doc_id)
            self.save()

    def add_inserted_batch(self, batch_id: int):
        """Mark a batch as inserted."""
        if batch_id not in self.data["inserted_batches"]:
            self.data["inserted_batches"].append(batch_id)
            self.save()