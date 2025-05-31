FROM python:3.9-slim

# Step 1: Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Set working directory
WORKDIR /app

# Step 3: Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy all code
COPY . .

CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501 --server.headless true"]
