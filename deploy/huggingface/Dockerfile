# Aegis web app for a free Hugging Face Space (Docker SDK, CPU basic).
# Build context: the repository root. Trained models go in deploy/huggingface/models/
# (bigvul.pt, juliet-cpp.pt, juliet-java.pt and their .sha256 files) before pushing.
FROM eclipse-temurin:21-jre-jammy

ARG JOERN_VERSION=v4.0.597
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv unzip curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/joern.zip \
       https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli-linux-x86_64.zip \
    && unzip -q /tmp/joern.zip -d /opt && rm /tmp/joern.zip
ENV PATH="/opt/joern-cli:/opt/venv/bin:${PATH}"

RUN python3 -m venv /opt/venv && pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY .streamlit ./.streamlit
RUN pip install --no-cache-dir -e ".[ui]" semgrep

# Hugging Face runs the container as user 1000.
RUN useradd -m -u 1000 aegis
COPY --chown=aegis deploy/huggingface/models /home/aegis/.aegis/models
USER aegis
ENV HOME=/home/aegis AEGIS_MODELS_DIR=/home/aegis/.aegis/models HF_HOME=/home/aegis/.cache/huggingface
# Download CodeBERT at build time so the first scan does not wait for it.
RUN python -c "from aegis.embeddings.codebert_embedder import load_model; load_model('cpu')"

EXPOSE 7860
CMD ["streamlit", "run", "src/aegis/ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true"]
