# syntax=docker/dockerfile:1.7
# DepthWizard2 model service (docker/model/app.py): DAv2-Small + Method 6 on ONNX Runtime, CPU.
# The weights come in through the "weights" build context (build.sh downloads and checks them); they are never in git.
#
#   docker buildx build --platform linux/amd64 --build-context weights=DIR -t IMAGE docker/model
FROM python:3.11-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY --from=weights dav2_small.onnx method6_full_dfc2019_hb_seed42.onnx /models/
COPY app.py /app/app.py
WORKDIR /app
RUN useradd --uid 10001 --no-create-home app
USER app
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
