FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (matches the repo's torch==2.11.0 pin, avoids the CUDA wheel)
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html ./

# writable caches for the HF-Spaces sandbox user
ENV HF_HOME=/tmp/hf
ENV OMP_NUM_THREADS=2

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
