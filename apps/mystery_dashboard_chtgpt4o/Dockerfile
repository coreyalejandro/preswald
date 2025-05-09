FROM python:3.12-slim

WORKDIR /app

# Install nodejs and npm (needed for frontend)
RUN apt-get update && apt-get install -y nodejs npm

# Copy pyproject.toml
COPY pyproject.toml .

# Install dependencies from pyproject.toml
RUN pip install .

# preswald dependency is declared in pyproject.toml and installed by pip install .

# Copy app and assets
COPY . .

EXPOSE 8501

ENV PYTHONPATH=/app
# Entrypoint script defined in preswald.toml
ENV SCRIPT_PATH=/app/hello.py
ENV PORT=8501

# Use startup script that calls start_server
CMD ["python", "run.py"]