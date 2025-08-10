# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Ensure venv's bin directory is in PATH
ENV PATH="/opt/venv/bin:$PATH"

# Install uv in the venv
RUN pip install uv

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies with uv inside the venv
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command to run the app
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
