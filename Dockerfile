# Base image: a slim Python 3.11 environment (small download, still full Python).
# Doesn't need to match your local Python version - the container brings its own.
FROM python:3.11-slim

# Prevents Python from buffering print()/input() output - important so your
# chat prompts ("You: ") actually show up immediately in an interactive terminal.
ENV PYTHONUNBUFFERED=1

# All commands below run from /app INSIDE the container (not your real machine)
WORKDIR /app

# Copy requirements FIRST, before the rest of the code. Docker caches each
# step - as long as requirements.txt hasn't changed, Docker reuses the
# cached "pip install" step instead of re-downloading everything (including
# torch/sentence-transformers, which are large) every time you rebuild after
# a small code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# Default command when the container starts - runs your normal chat agent.
# Overridden at run-time for the eval commands (see README).
CMD ["python", "main.py"]