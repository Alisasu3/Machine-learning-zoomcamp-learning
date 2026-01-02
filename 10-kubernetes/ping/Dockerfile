FROM python:3.9-slim

WORKDIR /app
COPY ping.py .

RUN pip install --no-cache-dir flask gunicorn

EXPOSE 9696
CMD ["gunicorn", "--bind=0.0.0.0:9696", "ping:app"]


# FROM python:3.9-slim

# RUN pip install pipenv

# WORKDIR /app

# COPY ["Pipfile", "Pipfile.lock", "./"]

# RUN pipenv install --system --deploy

# COPY "ping.py" .

# EXPOSE 9696

# ENV PYTHONPATH=/app

# ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "ping:app"]