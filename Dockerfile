FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY alembic.ini ./
COPY src ./src
COPY migrations ./migrations
COPY verticals ./verticals
RUN pip install --no-cache-dir .[postgres]
COPY config.example.yml ./config.example.yml
EXPOSE 8000
