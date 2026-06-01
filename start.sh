#!/bin/bash
set -e

python -c "
import sys
sys.path.insert(0, '/app')
from app.rag import get_collection_stats
stats = get_collection_stats()
count = stats.get('total_chunks', 0)
print(f'Vector store has {count} chunks.')
sys.exit(0 if count > 0 else 1)
" && echo "Already ingested, skipping." || {
    echo "Running ingestion..."
    python -c "
import sys
sys.path.insert(0, '/app')
from app.rag import ingest_documents
result = ingest_documents()
print(f'Done: {result[\"files_processed\"]} files, {result[\"chunks_stored\"]} chunks.')
"
}

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1