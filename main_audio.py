"""
Liquid AI LFM2-Audio-1.5B Model Inference (CPU-only)
Direct inference with liquid-audio library
"""

import argparse
import os
from pathlib import Path


def run_text_inference(text: str, max_tokens: int = 256):
    """Run text-only inference"""
    import torch
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
    
    print("Loading LFM2-Audio model (text mode)...")
    
    # Check for local model
    local_model_path = "/models/LFM2-Audio-1.5B"
    if os.path.exists(local_model_path):
        model_id = local_model_path
        print(f"Using local model: {model_id}")
    elif os.path.exists("LFM2-Audio-1.5B"):
        model_id = "LFM2-Audio-1.5B"
        print(f"Using local model: {model_id}")
    else:
        model_id = "LiquidAI/LFM2-Audio-1.5B"
        print(f"Using HuggingFace model: {model_id}")
    
    # Load model
    print("Loading processor and model... (this may take 2-3 minutes)")
    processor = LFM2AudioProcessor.from_pretrained(model_id).eval()
    model = LFM2AudioModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()
    print("✅ Model loaded successfully on CPU")
    
    # Set up chat
    chat = ChatState(processor)
    
    chat.new_turn("system")
    chat.add_text("You are a helpful assistant.")
    chat.end_turn()
    
    chat.new_turn("user")
    chat.add_text(text)
    chat.end_turn()
    
    chat.new_turn("assistant")
    
    print(f"\nPrompt: {text}\n")
    print("Generating response...")
    print("="*80)
    
    # Generate (sequential mode for text-only)
    full_text = ""
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=max_tokens):
            if t.numel() == 1:  # Text token
                decoded = processor.text.decode(t)
                print(decoded, end="", flush=True)
                full_text += decoded
    
    print("\n" + "="*80)


def run_tts(text: str, output_path: str = "output.wav"):
    """Run text-to-speech inference"""
    import torch
    import torchaudio
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
    
    print("Loading LFM2-Audio model (TTS mode)...")
    
    # Check for local model
    local_model_path = "/models/LFM2-Audio-1.5B"
    if os.path.exists(local_model_path):
        model_id = local_model_path
    elif os.path.exists("LFM2-Audio-1.5B"):
        model_id = "LFM2-Audio-1.5B"
    else:
        model_id = "LiquidAI/LFM2-Audio-1.5B"
    
    processor = LFM2AudioProcessor.from_pretrained(model_id).eval()
    model = LFM2AudioModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()
    print("✅ Model loaded successfully")
    
    # Set up chat
    chat = ChatState(processor)
    
    chat.new_turn("system")
    chat.add_text("Generate speech for the following text.")
    chat.end_turn()
    
    chat.new_turn("user")
    chat.add_text(text)
    chat.end_turn()
    
    chat.new_turn("assistant")
    
    print(f"\nGenerating speech for: {text}")
    print("Processing...")
    
    # Generate audio
    audio_out = []
    with torch.no_grad():
        for t in model.generate_sequential(
            **chat,
            max_new_tokens=512,
            audio_temperature=1.0,
            audio_top_k=4
        ):
            if t.numel() > 1:  # Audio tokens
                audio_out.append(t)
    
    if not audio_out:
        print("❌ No audio generated")
        return
    
    # Detokenize audio
    mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
    with torch.no_grad():
        waveform = processor.mimi.decode(mimi_codes)[0]
    
    # Save audio
    torchaudio.save(output_path, waveform.cpu(), 24000)
    print(f"✅ Audio saved to: {output_path}")
    print(f"   Sample rate: 24kHz")
    print(f"   Duration: {waveform.shape[1] / 24000:.2f}s")


def run_asr(audio_path: str):
    """Run automatic speech recognition"""
    import torch
    import torchaudio
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print("Loading LFM2-Audio model (ASR mode)...")
    
    # Check for local model
    local_model_path = "/models/LFM2-Audio-1.5B"
    if os.path.exists(local_model_path):
        model_id = local_model_path
    elif os.path.exists("LFM2-Audio-1.5B"):
        model_id = "LFM2-Audio-1.5B"
    else:
        model_id = "LiquidAI/LFM2-Audio-1.5B"
    
    processor = LFM2AudioProcessor.from_pretrained(model_id).eval()
    model = LFM2AudioModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).eval()
    print("✅ Model loaded successfully")
    
    # Load audio
    print(f"\nLoading audio: {audio_path}")
    wav, sampling_rate = torchaudio.load(audio_path)
    print(f"Sample rate: {sampling_rate}Hz, Duration: {wav.shape[1] / sampling_rate:.2f}s")
    
    # Set up chat
    chat = ChatState(processor)
    
    chat.new_turn("system")
    chat.add_text("Transcribe the following audio.")
    chat.end_turn()
    
    chat.new_turn("user")
    chat.add_audio(wav, sampling_rate)
    chat.end_turn()
    
    chat.new_turn("assistant")
    
    print("\nTranscribing...")
    print("="*80)
    
    # Generate transcription
    full_text = ""
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=512):
            if t.numel() == 1:  # Text token
                decoded = processor.text.decode(t)
                print(decoded, end="", flush=True)
                full_text += decoded
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run Liquid AI LFM2-Audio-1.5B model inference"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "tts", "asr", "demo"],
        default="demo",
        help="Inference mode: text (text-only), tts (text-to-speech), asr (speech-to-text), demo (run Gradio demo)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Text prompt (for text and tts modes)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="input.wav",
        help="Input audio file path (for asr mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path (for tts mode)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Liquid AI LFM2-Audio-1.5B - CPU Inference")
    print("="*80)
    
    if args.mode == "text":
        run_text_inference(args.prompt, args.max_tokens)
    
    elif args.mode == "tts":
        run_tts(args.prompt, args.output)
    
    elif args.mode == "asr":
        run_asr(args.audio)
    
    elif args.mode == "demo":
        print("\nStarting Gradio demo interface...")
        print("This will launch a web interface on http://localhost:7860/")
        print("\nTo start the demo, run:")
        print("  liquid-audio-demo")
        print("\nOr install demo dependencies first:")
        print("  pip install 'liquid-audio[demo]'")
        print("  liquid-audio-demo")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()