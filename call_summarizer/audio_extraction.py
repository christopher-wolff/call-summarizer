"""Audio extraction utilities for converting video files to audio."""

import os
import subprocess
from pathlib import Path
from typing import Optional


def extract_audio_from_video(video_path: str, output_path: str, audio_format: str = "wav") -> bool:
    """Extract audio from a video file using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        output_path: Path where the extracted audio should be saved
        audio_format: Audio format for output (default: wav)
        
    Returns:
        True if extraction was successful, False otherwise
        
    Raises:
        FileNotFoundError: If the input video file doesn't exist
        ValueError: If the video path or output path is invalid
    """
    video_path_obj = Path(video_path)
    output_path_obj = Path(output_path)
    
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not video_path_obj.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    
    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = _build_ffmpeg_command(video_path, output_path, audio_format)
    
    result = _run_ffmpeg_command(cmd)
    return result == 0


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds, or None if unable to determine
    """
    audio_path_obj = Path(audio_path)
    
    if not audio_path_obj.exists():
        return None
    
    cmd = _build_ffprobe_command(audio_path)
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration_str = result.stdout.strip()
    return float(duration_str)


def _build_ffmpeg_command(video_path: str, output_path: str, audio_format: str) -> list[str]:
    """Build the ffmpeg command for audio extraction."""
    return [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le" if audio_format.lower() == "wav" else "libmp3lame",
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite output file
        output_path
    ]


def _build_ffprobe_command(audio_path: str) -> list[str]:
    """Build the ffprobe command for getting audio duration."""
    return [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path
    ]


def _run_ffmpeg_command(cmd: list[str]) -> int:
    """Run the ffmpeg command and return the exit code."""
    result = subprocess.run(cmd, check=True)
    return result.returncode
