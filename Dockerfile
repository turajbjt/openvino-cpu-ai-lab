# --- STAGE 1: BUILDER ---
FROM docker.io/openvino/ubuntu22_dev:2024.0.0 AS builder

USER root
WORKDIR /build

# Added libncurses for terminal-related python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl gcc g++ python3-dev libncurses5-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install as a normal user to /install
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --prefix=/install -r requirements.txt

# --- STAGE 2: RUNTIME ---
FROM docker.io/openvino/ubuntu22_runtime:2024.0.0

USER root
WORKDIR /app

# Copy the installed packages from the builder prefix
COPY --from=builder /install /usr/local
COPY . .

# Fedora 43 / SELinux safety: ensure permissions are open for the app
RUN chmod -R 755 /app

WORKDIR /app
COPY . .
CMD ["python3", "main.py"]
