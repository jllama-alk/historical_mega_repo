FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Ollama is bundled in this same container (not a separate service) so that
# main.py's hardcoded "http://localhost:11434" default just works unmodified.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "bitsandbytes>=0.43.0" \
    "peft>=0.10.0" \
    "accelerate>=0.29.0" \
    langchain-core \
    langchain-chroma \
    langchain-ollama \
    rich

# Same absolute path as the host repo: rag/infastructure/vectordb/vector.py
# hardcodes FD_FOLDERS as an absolute path, so this avoids touching that file.
WORKDIR /mnt/linux_storage/projects/Historical_AI
COPY . .

# main.py does `from vectordb.vector import HistoricalVector`; the real
# package lives under rag/infastructure, so it must be on the path.
ENV PYTHONPATH=/mnt/linux_storage/projects/Historical_AI/rag/infastructure

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

WORKDIR /mnt/linux_storage/projects/Historical_AI/main
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "main.py"]
