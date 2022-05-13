from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return "Service is online."

from transformers import pipeline
# sentiment_model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


# sentiment_query_sentence = get_random_comment(top_comments)
# sentiment = sentiment_model(sentiment_query_sentence)
# print(f"Sentiment test: {sentiment_query_sentence} == {sentiment})

from pydantic import BaseModel

class PredictionRequest(BaseModel):
  query_string: str

@app.post("/my-endpoint")
def my_endpoint(request: PredictionRequest):
    return sentiment_model(request.query_string)
