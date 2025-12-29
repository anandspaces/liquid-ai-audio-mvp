"""
Client for Liquid AI LFM2-Audio-1.5B API
Supports text, audio, and multimodal requests
"""

import requests
import json
import base64
from typing import List, Dict, Optional
from pathlib import Path


class LiquidAudioClient:
    """Client for interacting with LFM2-Audio API"""
    
    def __init__(self, base_url: str = "http://localhost:8091"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4
    ) -> Dict:
        """
        Send chat completion request with text
        Returns text and optional audio in response
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Text sampling temperature
            audio_temperature: Audio sampling temperature
            audio_top_k: Audio top-k sampling
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "audio_temperature": audio_temperature,
            "audio_top_k": audio_top_k
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def text_to_speech(
        self,
        text: str,
        output_path: str,
        max_tokens: int = 512,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4
    ) -> str:
        """
        Convert text to speech and save to file
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            max_tokens: Maximum tokens to generate
            audio_temperature: Audio sampling temperature
            audio_top_k: Audio top-k sampling
            
        Returns:
            Path to saved audio file
        """
        payload = {
            "text": text,
            "max_tokens": max_tokens,
            "audio_temperature": audio_temperature,
            "audio_top_k": audio_top_k
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload
        )
        response.raise_for_status()
        
        # Save audio file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path
    
    def speech_to_text(
        self,
        audio_path: str,
        max_tokens: int = 512
    ) -> str:
        """
        Convert speech to text (ASR/STT)
        
        Args:
            audio_path: Path to audio file
            max_tokens: Maximum tokens to generate
            
        Returns:
            Transcribed text
        """
        with open(audio_path, 'rb') as f:
            files = {'file': (Path(audio_path).name, f, 'audio/wav')}
            data = {'max_tokens': max_tokens}
            
            response = self.session.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=data
            )
        
        response.raise_for_status()
        return response.json()['text']
    
    def speech_to_speech(
        self,
        audio_path: str,
        output_path: str,
        max_tokens: int = 512,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4
    ) -> tuple[str, str]:
        """
        Speech-to-speech conversation
        
        Args:
            audio_path: Path to input audio file
            output_path: Where to save response audio
            max_tokens: Maximum tokens to generate
            audio_temperature: Audio sampling temperature
            audio_top_k: Audio top-k sampling
            
        Returns:
            Tuple of (output_path, transcription)
        """
        with open(audio_path, 'rb') as f:
            files = {'file': (Path(audio_path).name, f, 'audio/wav')}
            data = {
                'max_tokens': max_tokens,
                'audio_temperature': audio_temperature,
                'audio_top_k': audio_top_k
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/audio/chat",
                files=files,
                data=data
            )
        
        response.raise_for_status()
        
        # Save response audio
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Get transcription from header
        transcription = response.headers.get('X-Transcription', '')
        
        return output_path, transcription


def main():
    """Example usage"""
    print("="*80)
    print("Liquid AI LFM2-Audio-1.5B API Client Examples")
    print("="*80)
    
    client = LiquidAudioClient()
    
    # Health check
    print("\n1. Health Check")
    print("-" * 40)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Model loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Chat completion (text-to-text with optional audio)
    print("\n2. Chat Completion (Text)")
    print("-" * 40)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=256
        )
        print(f"Response: {response['choices'][0]['message']['content']}")
        
        # Check if audio was generated
        if response['choices'][0]['message'].get('audio'):
            print("✅ Audio response available (base64 encoded)")
    except Exception as e:
        print(f"Error: {e}")
    
    # Text-to-Speech
    print("\n3. Text-to-Speech")
    print("-" * 40)
    try:
        output_file = "output_speech.wav"
        client.text_to_speech(
            text="Hello, this is a test of the text to speech system.",
            output_path=output_file
        )
        print(f"✅ Speech saved to: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Speech-to-Text (if you have an audio file)
    print("\n4. Speech-to-Text")
    print("-" * 40)
    input_audio = "input_audio.wav"
    if Path(input_audio).exists():
        try:
            transcription = client.speech_to_text(input_audio)
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Skipping - no audio file at {input_audio}")
    
    # Speech-to-Speech (if you have an audio file)
    print("\n5. Speech-to-Speech")
    print("-" * 40)
    if Path(input_audio).exists():
        try:
            output_file, transcription = client.speech_to_speech(
                audio_path=input_audio,
                output_path="response_audio.wav"
            )
            print(f"✅ Response saved to: {output_file}")
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Skipping - no audio file at {input_audio}")
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)


if __name__ == "__main__":
    main()