"""Pydantic models for the call summarizer project."""

import typing
import pydantic


class TranscriptSegment(pydantic.BaseModel):
    """A single segment of a transcript with timing information."""
    
    id: int = pydantic.Field(description="Unique identifier for this segment")
    start: float = pydantic.Field(description="Start time in seconds")
    end: float = pydantic.Field(description="End time in seconds")
    text: str = pydantic.Field(description="Transcribed text for this segment")


class Transcript(pydantic.BaseModel):
    """Complete transcript data from OpenAI Whisper API."""
    
    text: str = pydantic.Field(description="Full transcribed text")
    language: str = pydantic.Field(description="Detected language")
    duration: float = pydantic.Field(description="Total duration in seconds")
    segments: typing.List[TranscriptSegment] = pydantic.Field(
        description="List of transcript segments with timing information"
    )


class AudioExtractionResult(pydantic.BaseModel):
    """Result of audio extraction from video file."""
    
    success: bool = pydantic.Field(description="Whether extraction was successful")
    input_path: str = pydantic.Field(description="Path to input video file")
    output_path: str = pydantic.Field(description="Path to extracted audio file")
    duration: typing.Optional[float] = pydantic.Field(
        default=None, 
        description="Duration of extracted audio in seconds"
    )
    error_message: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Error message if extraction failed"
    )


class TranscriptionResult(pydantic.BaseModel):
    """Result of audio transcription."""
    
    success: bool = pydantic.Field(description="Whether transcription was successful")
    input_path: str = pydantic.Field(description="Path to input audio file")
    output_path: str = pydantic.Field(description="Path to transcript JSON file")
    transcript: typing.Optional[Transcript] = pydantic.Field(
        default=None,
        description="Transcribed content if successful"
    )
    error_message: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Error message if transcription failed"
    )