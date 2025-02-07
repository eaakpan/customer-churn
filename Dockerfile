FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./src ./src
COPY ./database ./database
COPY ./modeling ./modeling
COPY ./helpers ./helpers
COPY ./config ./config
COPY runtime_data ./data
COPY ./app.py ./src/app.py

ENV PYTHONPATH /code
ENV PYTHONUNBUFFERED 1

EXPOSE 8000

CMD ["gunicorn"  , "-b", "0.0.0.0:8000", "src.app:server"]