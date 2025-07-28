# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Set Python path
ENV PYTHONPATH=/app

# Train the model during build (optional - for demo purposes)
# In production, you might want to mount models as volumes
RUN python src/train.py

# Default command runs the prediction script
CMD ["python", "src/predict.py"]