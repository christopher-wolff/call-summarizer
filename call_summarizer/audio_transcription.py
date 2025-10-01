"""Audio transcription utilities for converting audio files to text."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import openai

from .models import Transcript, TranscriptSegment


def transcribe_audio_file(audio_path: str, output_path: str, model: str = "whisper-1") -> bool:
    """Transcribe an audio file using OpenAI's Whisper API.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path where the transcription should be saved (JSON format)
        model: OpenAI Whisper model to use (default: whisper-1)
        
    Returns:
        True if transcription was successful, False otherwise
        
    Raises:
        FileNotFoundError: If the input audio file doesn't exist
        ValueError: If the audio path or output path is invalid
    """
    audio_path_obj = Path(audio_path)
    output_path_obj = Path(output_path)
    
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not audio_path_obj.is_file():
        raise ValueError(f"Path is not a file: {audio_path}")
    
    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Get transcription from OpenAI
    transcript = _get_whisper_transcription(audio_path, model)
    
    if transcript is None:
        return False
    
    # Save transcription to file
    _save_transcription(transcript, output_path)
    
    return True


def get_transcription_text(transcription_path: str) -> str:
    """Get the transcribed text from a transcription file.
    
    Args:
        transcription_path: Path to the transcription JSON file
        
    Returns:
        The transcribed text
        
    Raises:
        FileNotFoundError: If the transcription file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(transcription_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('text', '')


def get_transcription_metadata(transcription_path: str) -> Dict[str, Any]:
    """Get metadata from a transcription file.
    
    Args:
        transcription_path: Path to the transcription JSON file
        
    Returns:
        Dictionary containing metadata
        
    Raises:
        FileNotFoundError: If the transcription file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(transcription_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
            'duration': data.get('duration'),
            'language': data.get('language'),
            'segments': data.get('segments', [])
        }


def load_transcript(transcription_path: str) -> Transcript:
    """Load a transcript file and return a Transcript Pydantic model.
    
    Args:
        transcription_path: Path to the transcription JSON file
        
    Returns:
        Transcript Pydantic model
        
    Raises:
        FileNotFoundError: If the transcription file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValidationError: If the JSON doesn't match the expected schema
    """
    with open(transcription_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Transcript.model_validate(data)


def _get_whisper_transcription(audio_path: str, model: str) -> Optional[Transcript]:
    """Get transcription from OpenAI Whisper API."""
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    with open(audio_path, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json"
        )
    
    # Convert to Pydantic models
    segments = []
    if hasattr(transcript, 'segments'):
        segments = [
            TranscriptSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text
            )
            for segment in transcript.segments
        ]
    
    return Transcript(
        text=transcript.text,
        language=transcript.language,
        duration=transcript.duration,
        segments=segments
    )


def _save_transcription(transcript: Transcript, output_path: str) -> None:
    """Save transcription data to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcript.model_dump(), f, indent=2, ensure_ascii=False)
