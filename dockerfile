FROM python:3.12-slim

WORKDIR /app

COPY ProductionFiles/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# CMD ["python3", "app.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]