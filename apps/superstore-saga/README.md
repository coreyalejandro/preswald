# Superstore Saga

This is a Preswald application that visualizes the Superstore sample dataset.

## Prerequisites

- Python 3.8 or higher
- Docker (optional)

## Local Development

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   preswald run
   ```
   The app will be available at http://localhost:8502.

## Docker

Build and run with Docker:
```bash
docker build -t superstore-saga .
docker run -p 8502:8502 superstore-saga
```
The app will be available at http://localhost:8502.