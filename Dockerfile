FROM python:3.11-slim

# Create user to run the app (Hugging Face Spaces requirement for Docker)
RUN useradd -m -u 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN chown user:user $HOME/app

# Switch to non-root user
USER user

# Install dependencies first to cache the layer
COPY --chown=user:user requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files
COPY --chown=user:user . .

# Expose port 7860 for HF Spaces
EXPOSE 7860

# Run the FastAPI server via main.py
CMD ["python", "main.py"]
