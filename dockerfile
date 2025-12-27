# Use a specific slim version for reproducibility and size
FROM python:3.12-slim

# Set environment variables to prevent Python from buffering stdout/stderr
# and to prevent writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install Dependencies First (Better Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Pre-download NLTK data (Stopwords, Wordnet, OMW)
# This prevents the app from trying to download them at runtime
RUN python -m nltk.downloader stopwords wordnet omw-1.4

# 3. Create a fake .git directory
# Your code uses 'from_root', which looks for a .git folder to find the root.
# Without this, the app might crash inside Docker.
RUN mkdir -p /app/.git

# 4. Copy Only Necessary Directories
# We copy explicitly to maintain structure
COPY src/ /app/src/
COPY models/ /app/models/
COPY templates/ /app/templates/
COPY app.py /app/

# Expose the port
EXPOSE 5000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]