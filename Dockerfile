# Backend + built dashboard in one image. CPU PyTorch by default for portability;
# for NVIDIA GPUs build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
# and run with `--gpus all`.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "psycopg[binary]>=3.2"
COPY backend/ backend/
COPY configs/ configs/
COPY scripts/ scripts/
COPY main.py .
COPY --from=web /web/dist frontend/dist
# model + data are mounted as volumes (never baked into the image / sent to browsers)
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
