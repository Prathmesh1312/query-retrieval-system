# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for document parsing (if using unstructured)
# RUN apt-get update && apt-get install -y \
#     poppler-utils \
#     tesseract-ocr \
#     libreoffice \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final - Create the lean production image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application code
COPY main.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# For production, consider using Gunicorn with Uvicorn workers for better concurrency:
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
