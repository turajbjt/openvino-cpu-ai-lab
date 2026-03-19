# Use official OpenVINO Ubuntu 22.04 base image
FROM openvino/ubuntu22_dev:2024.0.0

# Set user to root for installation
USER root

# Install basic development utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/lab

# Set environmental variable to prioritize CPU during optimization
ENV OV_FRONTEND_PREFER_CPU=1

# By default, drop into a bash shell
CMD ["/bin/bash"]
