FROM python:3.11-slim

# to prevent python from writing .pyc files (saves tiny space)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 3. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy code
COPY . .

CMD ["sh", "-c", "gunicorn -w 1 --threads 8 --timeout 0 -b 0.0.0.0:${PORT} app:app"]