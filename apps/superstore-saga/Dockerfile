FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the Preswald default port
EXPOSE 8502

# Start the Preswald app
CMD ["preswald", "run", "--host", "0.0.0.0"]