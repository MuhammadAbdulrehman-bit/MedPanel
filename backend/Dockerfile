# Use Python 3.10 slim base
FROM python:3.10-slim

# Set working directory
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

# Install Python dependencies
# CPU-only torch inside Docker (GPU not needed in cloud)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .
# Copy model weights
COPY backend/models/skin_efficientnet_best.pth /app/backend/models/
COPY backend/models/retinal_v2_best.pth /app/backend/models/
# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]