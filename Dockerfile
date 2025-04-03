# Artificial Traders v4/Multi_Ai/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies and update setuptools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir setuptools>=61.0.0

# Install TA-Lib C library from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Verify TA-Lib library installation
RUN ls -l /usr/lib/libta_lib.so || echo "TA-Lib C library not found"

# Set environment variables for TA-Lib
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include

COPY requirements.txt .
# Install Python dependencies step-by-step with debug
RUN pip3 install --no-cache-dir numpy && \
    pip3 install --no-cache-dir TA-Lib && \
    pip3 install --no-cache-dir -r requirements.txt || echo "Failed to install requirements.txt"

COPY .. .

ENV PYTHONPATH=/app/src

EXPOSE 5000

CMD ["bash", "start.sh"]