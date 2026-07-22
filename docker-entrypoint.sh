#!/bin/bash
set -e

ollama serve &

until curl -sf http://localhost:11434 >/dev/null; do
    sleep 1
done

# langchain_ollama does NOT auto-pull missing models — it just errors, so
# these must exist before main.py touches OllamaLLM/OllamaEmbeddings.
ollama pull gemma4:e4b
ollama pull embeddinggemma:latest

exec "$@"
