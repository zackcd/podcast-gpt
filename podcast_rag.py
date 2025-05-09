import os
from typing import List
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

class PodcastRAG:
    def __init__(self, transcripts_dir: str = 'data/transcripts'):
        self.transcripts_dir = transcripts_dir
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        
        # Connect to Milvus
        connections.connect(host='localhost', port='19530')
        self._setup_collection()
        self._load_transcripts()
        
    def _setup_collection(self):
        collection_name = "podcast_chunks"
        
        if utility.exists_collection(collection_name):
            self.collection = Collection(collection_name)
            return
            
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields=fields)
        
        # Create collection
        self.collection = Collection(name=collection_name, schema=schema)
        
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        
    def _load_transcripts(self):
        """Load and chunk all podcast transcripts into Milvus"""
        self.collection.load()
        
        # Skip if collection is not empty
        if self.collection.num_entities > 0:
            return
            
        chunks = []
        embeddings = []
        
        for filename in os.listdir(self.transcripts_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.transcripts_dir, filename)
                with open(filepath, 'r') as f:
                    text = f.read()
                    # Split into dialogue chunks
                    exchanges = text.split('\n')
                    chunk_size = 5
                    text_chunks = ['\n'.join(exchanges[i:i+chunk_size]) 
                                 for i in range(0, len(exchanges), chunk_size//2)]
                    chunks.extend(text_chunks)
                    
        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = self.encoder.encode(batch)
            
            # Insert into Milvus
            entities = [
                [j for j in range(i, min(i+batch_size, len(chunks)))],  # id
                batch,  # text
                batch_embeddings.tolist()  # embedding
            ]
            self.collection.insert(entities)
            
        self.collection.flush()
        
    def get_relevant_context(self, query: str, n_results: int = 50) -> List[str]:
        """Get most relevant transcript chunks using Milvus"""
        query_embedding = self.encoder.encode([query])[0]
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=n_results,
            output_fields=["text"]
        )
        
        return [hit.entity.get('text') for hit in results[0]]

    def augment_prompt(self, query: str) -> str:
        """Augment an LLM prompt with relevant podcast context"""
        context = self.get_relevant_context(query)
        augmented_prompt = (
            "Based on these conversations from the Technology Brothers podcast:\n\n" + 
            "\n---\n".join(context) +
            f"\n\nDiscuss this question in your typical podcast style: {query}"
        )
        return augmented_prompt
