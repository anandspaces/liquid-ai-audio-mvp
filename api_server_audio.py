"""
FastAPI server for Liquid AI LFM2-Audio-1.5B
Uses trust_remote_code to load custom model architecture
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import torch
import torchaudio
import logging
import os
import tempfile
import base64
from io import BytesIO
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model and processor
model = None
processor = None
MODEL_LOADED = False

# Request models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    audio_temperature: float = 1.0
    audio_top_k: int = 4

class TTSRequest(BaseModel):
    text: str
    max_tokens: int = 512
    audio_temperature: float = 1.0
    audio_top_k: int = 4


async def load_model():
    """Load model on startup"""
    global model, processor, MODEL_LOADED
    
    logger.info("="*80)
    logger.info("Loading Liquid AI LFM2-Audio-1.5B model...")
    logger.info("="*80)
    
    # Check for local model
    local_model_path = "/models/LFM2-Audio-1.5B"
    if os.path.exists(local_model_path):
        model_path = local_model_path
        logger.info(f"✅ Found local model: {model_path}")
    elif os.path.exists("LFM2-Audio-1.5B"):
        model_path = "LFM2-Audio-1.5B"
        logger.info(f"✅ Found local model: {model_path}")
    else:
        model_path = "LiquidAI/LFM2-Audio-1.5B"
        logger.info(f"📥 Will use HuggingFace: {model_path}")
    
    try:
        from transformers import AutoModel, AutoProcessor
        
        # Load processor with trust_remote_code
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        logger.info("✅ Processor loaded")
        
        # Load model with trust_remote_code
        logger.info("Loading model (2-3 minutes)...")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        ).eval()
        
        MODEL_LOADED = True
        logger.info("="*80)
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Device: CPU")
        logger.info(f"   Precision: bfloat16")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        MODEL_LOADED = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Liquid AI LFM2-Audio-1.5B API",
    description="Audio API for Liquid AI LFM2-Audio-1.5B model",
    version="2.1.0",
    lifespan=lifespan
)


def check_model_loaded():
    """Verify model is loaded"""
    if not MODEL_LOADED or model is None or processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Server may still be initializing."
        )


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "device": "cpu",
        "capabilities": [
            "text-to-text",
            "text-to-speech",
            "speech-to-text",
            "speech-to-speech"
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatRequest):
    """Chat completion endpoint"""
    check_model_loaded()
    
    try:
        # Build conversation string
        conversation = ""
        for msg in request.messages:
            conversation += f"{msg.role.title()}: {msg.content}\n\n"
        conversation += "Assistant: "
        
        # Process input
        inputs = processor(conversation, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode response
        response_text = processor.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        ).strip()
        
        return {
            "id": f"chatcmpl-{abs(hash(str(request.messages))) % 10**16}",
            "object": "chat.completion",
            "choices": [{
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest):
    """Text-to-speech generation"""
    check_model_loaded()
    
    try:
        logger.info(f"TTS request: {request.text[:50]}...")
        
        # Check if model has TTS capability
        if not hasattr(model, 'generate_speech'):
            raise HTTPException(
                status_code=501,
                detail="TTS not supported by this model configuration"
            )
        
        # Generate speech
        with torch.no_grad():
            audio_output = model.generate_speech(
                text=request.text,
                processor=processor,
                max_tokens=request.max_tokens
            )
        
        # Save to buffer
        buffer = BytesIO()
        torchaudio.save(buffer, audio_output.cpu(), 24000, format="wav")
        buffer.seek(0)
        
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/transcriptions")
async def speech_to_text(
    file: UploadFile = File(...),
    max_tokens: int = Form(512)
):
    """Speech-to-text transcription"""
    check_model_loaded()
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load audio
        wav, sr = torchaudio.load(tmp_path)
        logger.info(f"Loaded audio: {wav.shape}, {sr}Hz")
        
        # Check if model has ASR capability
        if not hasattr(model, 'transcribe'):
            raise HTTPException(
                status_code=501,
                detail="ASR not supported by this model configuration"
            )
        
        # Transcribe
        with torch.no_grad():
            transcription = model.transcribe(
                audio=wav,
                sample_rate=sr,
                processor=processor,
                max_tokens=max_tokens
            )
        
        return {"text": transcription.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/v1/audio/chat")
async def speech_to_speech(
    file: UploadFile = File(...),
    max_tokens: int = Form(512),
    audio_temperature: float = Form(1.0),
    audio_top_k: int = Form(4)
):
    """Speech-to-speech conversation"""
    check_model_loaded()
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load audio
        wav, sr = torchaudio.load(tmp_path)
        
        # Check if model has S2S capability
        if not hasattr(model, 'speech_to_speech'):
            raise HTTPException(
                status_code=501,
                detail="Speech-to-speech not supported by this model configuration"
            )
        
        # Process speech-to-speech
        with torch.no_grad():
            output_audio, transcription = model.speech_to_speech(
                audio=wav,
                sample_rate=sr,
                processor=processor,
                max_tokens=max_tokens
            )
        
        # Save response
        buffer = BytesIO()
        torchaudio.save(buffer, output_audio.cpu(), 24000, format="wav")
        buffer.seek(0)
        
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "X-Transcription": transcription.strip(),
                "Content-Disposition": "attachment; filename=response.wav"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"S2S error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091, log_level="info")