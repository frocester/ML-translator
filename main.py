from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
# import nest_asyncio
from uvicorn import run

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (you can restrict it to specific origins)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST requests
    allow_headers=["*"],  # Allow all headers
)

class TranslationRequest(BaseModel):
    text: str

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model_3/")

@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    input_text = request.text
    tokenized = tokenizer([input_text], return_tensors='np')
    out = model.generate(**tokenized, max_length=128)
    with tokenizer.as_target_tokenizer():
        translated_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"translated_text": translated_text}


# nest_asyncio.apply()
# run(app, host="0.0.0.0")