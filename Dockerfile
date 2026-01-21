FROM ubuntu:22.04
LABEL authors="gukhwan"
LABEL build_date="2026-01"

EXPOSE 8080

WORKDIR /app

RUN apt-get update

RUN apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

RUN uv init --python 3.12 .
RUN uv add python-dotenv fastapi===0.128.0 uvicorn==0.40.0 gunicorn==23.0.0 openai==2.15.0 langfuse==3.11.2

RUN mkdir -p /app/log/gunicorn && mkdir -p /var/run/gunicorn

#COPY . .
COPY /src /app
COPY logs /app
COPY .env /app

CMD ["tail", "-f", "/dev/null"]
