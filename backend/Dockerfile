FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY backend/requirements.txt .

# Install CPU-only torch
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Download model weights from HuggingFace at build time
RUN python -c "
from huggingface_hub import hf_hub_download
import shutil
import os

os.makedirs('backend/models', exist_ok=True)

print('Downloading skin model...')
path = hf_hub_download(
    repo_id='FHKHTHJKKL/mediscan-weights',
    filename='skin_efficientnet_best.pth',
    repo_type='model'
)
shutil.copy(path, 'backend/models/skin_efficientnet_best.pth')
print('Skin model ready')

print('Downloading retinal model...')
path = hf_hub_download(
    repo_id='FHKHTHJKKL/mediscan-weights',
    filename='retinal_v2_best.pth',
    repo_type='model'
)
shutil.copy(path, 'backend/models/retinal_v2_best.pth')
print('Retinal model ready')
"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]