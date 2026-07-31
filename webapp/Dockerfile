FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime-only deps (no torch / matplotlib / jupyter — the webapp doesn't need them).
COPY requirements-webapp.txt .
RUN pip install --no-cache-dir -r requirements-webapp.txt

COPY frd ./frd
COPY webapp ./webapp

EXPOSE 8055

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8055"]
