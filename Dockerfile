FROM python:3.10-slim

WORKDIR /app

# Avoid writing .pyc files and ensure logs flush immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (minimal; matplotlib works fine with manylinux wheels)
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Default command is overridden by docker-compose services.
CMD ["python", "-c", "print('Use docker-compose to run services')"]

