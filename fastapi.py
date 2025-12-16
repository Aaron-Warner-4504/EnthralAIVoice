import io
import torch
import soundfile as sf
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel, Field
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice TTS API",
    description="Streaming + MP3 endpoints for Indic TTS",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load model + tokenizers
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading model on device: {device}")

try:
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts"
    ).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# -------------------------
# Voice Map
# -------------------------
VOICE_MAP = {
    "ad-hi": {
        "name": "Aditi",
        "lang": "Hindi",
        "desc": "Aditi speaks with a higher pitch in a clear, close environment."
    },
    "di-hi": {
        "name": "Divya",
        "lang": "Hindi",
        "desc": "Divya's voice is balanced, neutral, with close high-quality recording."
    },
    "ro-hi": {
        "name": "Rohit",
        "lang": "Hindi",
        "desc": "Rohit speaks with a warm, medium-pitched, natural tone."
    },
    "su-mr": {
        "name": "Sunita",
        "lang": "Marathi",
        "desc": "Sunita speaks with a high pitch and expressive tone in Marathi."
    },
    "sa-mr": {
        "name": "Sanjay",
        "lang": "Marathi",
        "desc": "Sanjay speaks with a deep and clear voice in Marathi."
    },
    "ad-en": {
        "name": "Aditi",
        "lang": "English",
        "desc": "Aditi speaks fluent Indian English with a clear neutral tone."
    }
}

# -------------------------
# Request Models
# -------------------------
class TTSRequest(BaseModel):
    voice_id: str = Field(..., description="Voice ID from available voices")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")

# -------------------------
# Helper Functions
# -------------------------
def generate_audio(voice_id: str, text: str):
    """Generate audio from text using specified voice."""
    if voice_id not in VOICE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice_id. Available: {list(VOICE_MAP.keys())}"
        )
    
    try:
        description = VOICE_MAP[voice_id]["desc"]
        
        desc_ids = desc_tokenizer(description, return_tensors="pt").to(device)
        prompt_ids = prompt_tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():  # Save memory
            out = model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
            )
        
        audio_arr = out.cpu().numpy().squeeze()
        return audio_arr, model.config.sampling_rate
    
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation error: {str(e)}")

# -------------------------
# Endpoints
# -------------------------
@app.post("/tts/stream", summary="Stream audio as WAV")
async def tts_stream(req: TTSRequest):
    """Generate and stream audio in WAV format."""
    logger.info(f"Stream request: voice={req.voice_id}, text_len={len(req.text)}")
    
    audio_arr, sr = generate_audio(req.voice_id, req.text)
    
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_arr, sr, format="WAV")
    wav_buffer.seek(0)
    
    return StreamingResponse(
        wav_buffer,
        media_type="audio/wav",
        headers={
            "Content-Length": str(wav_buffer.getbuffer().nbytes),
            "Accept-Ranges": "bytes"
        }
    )

@app.post("/tts/mp3", summary="Download audio as MP3")
async def tts_mp3(req: TTSRequest):
    """Generate and download audio in MP3 format."""
    logger.info(f"MP3 request: voice={req.voice_id}, text_len={len(req.text)}")
    
    audio_arr, sr = generate_audio(req.voice_id, req.text)
    
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_arr, sr, format="WAV")
    wav_buffer.seek(0)
    
    audio_seg = AudioSegment.from_wav(wav_buffer)
    mp3_buffer = io.BytesIO()
    audio_seg.export(mp3_buffer, format="mp3", bitrate="192k")  # Specify bitrate
    mp3_buffer.seek(0)
    
    return Response(
        content=mp3_buffer.read(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{req.voice_id}.mp3"'
        }
    )

@app.get("/voices", summary="List available voices")
async def list_voices():
    """Get all available voice configurations."""
    return {
        "count": len(VOICE_MAP),
        "voices": [
            {
                "voice_id": vid,
                "name": data["name"],
                "language": data["lang"],
                "description": data["desc"]
            }
            for vid, data in VOICE_MAP.items()
        ]
    }

@app.get("/health", summary="Health check")
def health():
    """Service health and device information."""
    return {
        "status": "ok",
        "device": device,
        "model_loaded": model is not None,
        "available_voices": len(VOICE_MAP)
    }

@app.get("/", summary="API Information")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Kokoro Voice TTS API",
        "version": "1.0",
        "endpoints": {
            "docs": "/docs",
            "voices": "/voices",
            "stream": "/tts/stream",
            "mp3": "/tts/mp3"
        }
    }
