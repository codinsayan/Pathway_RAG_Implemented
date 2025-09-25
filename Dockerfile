# Use a Python 3.12 base image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    supervisor \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# --- Install Dependencies (Cached Layer) ---
# 1. Install Python dependencies (including gunicorn)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/backend/requirements.txt

# 2. Install Node.js dependencies
COPY frontend/package.json frontend/package-lock.json* /app/frontend/
RUN npm --prefix /app/frontend install
# Install a simple static server for the frontend
RUN npm install -g http-server

# --- Copy Application Code ---
COPY backend /app/backend
COPY frontend /app/frontend

# --- Frontend Build ---
# No need to set placeholder ENV vars here anymore
RUN npm --prefix /app/frontend run build

# --- Final Setup ---
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8080 8000

# The command to start the application
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]