FROM python:3.10-slim

# Install system dependencies (e.g. libgomp1 for ONNX Runtime OpenMP support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Avoid writing .pyc files and ensure logs flush immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user

# Create a non-root user with UID 1000 (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

# Pre-create Streamlit configuration directory in home
RUN mkdir -p /home/user/.streamlit

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy all source files and set appropriate ownership
COPY --chown=user:user . /app

# Expose port 7860 (Hugging Face Spaces default web port)
EXPOSE 7860

# Default command to run our unified process manager
CMD ["python", "start.py"]
