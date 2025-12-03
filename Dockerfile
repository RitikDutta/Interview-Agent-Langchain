# Use a small base
FROM python:3.11-slim

# Create non-root user
RUN useradd -m appuser

# Workdir
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Permissions & switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

# Env + port (Cloud Run also uses PORT)
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Gunicorn: 2 workers, gthread for I/O, bind to $PORT
CMD ["bash", "-lc", "gunicorn -w 2 -k gthread -b 0.0.0.0:${PORT} app:app"]
