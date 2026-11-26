# Aegis web app for a free Hugging Face Space (Docker SDK, CPU basic).
# Build context: the repository root. Trained models go in deploy/huggingface/models/
# (bigvul.pt, juliet-cpp.pt, juliet-java.pt and their .sha256 files) before pushing.
# Ubuntu 24.04 (noble) ships Python 3.12, which the pinned dependencies need.
FROM eclipse-temurin:21-jre-noble

ARG JOERN_VERSION=v4.0.597
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv unzip curl \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && JOERN_URL=https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli-linux-x86_64.zip \
    && curl -fsSL --http1.1 -o joern-cli-linux-x86_64.zip.sha512 "${JOERN_URL}.sha512" \
    && for i in 1 2 3 4 5 6; do \
         curl -fSL --http1.1 --retry 5 --retry-all-errors -C - -o joern-cli-linux-x86_64.zip "${JOERN_URL}" && break; \
         echo "download interrupted, resuming ($i)"; sleep 10; \
       done \
    && echo "$(cut -d' ' -f1 joern-cli-linux-x86_64.zip.sha512)  joern-cli-linux-x86_64.zip" | sha512sum -c - \
    && unzip -q joern-cli-linux-x86_64.zip -d /opt && rm joern-cli-linux-x86_64.zip*
ENV PATH="/opt/joern-cli:/opt/venv/bin:${PATH}"

RUN python3 -m venv /opt/venv && pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY .streamlit ./.streamlit
RUN pip install --no-cache-dir -e ".[ui]" semgrep

# Hugging Face runs the container as user 1000. The noble image's default "ubuntu" user
# already holds that ID, so it is removed first.
RUN (userdel -r ubuntu 2>/dev/null || true) && useradd -m -u 1000 aegis
COPY --chown=aegis deploy/huggingface/models /home/aegis/.aegis/models
USER aegis
ENV HOME=/home/aegis AEGIS_MODELS_DIR=/home/aegis/.aegis/models HF_HOME=/home/aegis/.cache/huggingface
# Download CodeBERT at build time so the first scan does not wait for it.
RUN python -c "from aegis.embeddings.codebert_embedder import load_model; load_model('cpu')"

EXPOSE 7860
CMD ["streamlit", "run", "src/aegis/ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true"]
