FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_DATA_DIR=/app/data
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY examples /app/examples
COPY .streamlit /app/.streamlit
RUN useradd -u 10001 -m rag && mkdir -p /app/data && chown -R rag:rag /app/data
USER rag
EXPOSE 8507
CMD ["sh", "-c", "python -m streamlit run src/asgi_app.py --server.address=0.0.0.0 --server.port=${PORT:-8507} --server.headless=true"]
