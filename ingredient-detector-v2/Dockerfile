FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-m", "src.data.config_utils"]
