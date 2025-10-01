"""Pydantic models for the call summarizer project."""

from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single segment of a transcript with timing information."""
    
    id: int = Field(description="Unique identifier for this segment")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text for this segment")


class Transcript(BaseModel):
    """Complete transcript data from OpenAI Whisper API."""
    
    text: str = Field(description="Full transcribed text")
    language: str = Field(description="Detected language")
    duration: float = Field(description="Total duration in seconds")
    segments: List[TranscriptSegment] = Field(
        description="List of transcript segments with timing information"
    )


class AudioExtractionResult(BaseModel):
    """Result of audio extraction from video file."""
    
    success: bool = Field(description="Whether extraction was successful")
    input_path: str = Field(description="Path to input video file")
    output_path: str = Field(description="Path to extracted audio file")
    duration: Optional[float] = Field(
        default=None, 
        description="Duration of extracted audio in seconds"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if extraction failed"
    )


class TranscriptionResult(BaseModel):
    """Result of audio transcription."""
    
    success: bool = Field(description="Whether transcription was successful")
    input_path: str = Field(description="Path to input audio file")
    output_path: str = Field(description="Path to transcript JSON file")
    transcript: Optional[Transcript] = Field(
        default=None,
        description="Transcribed content if successful"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if transcription failed"
    )
