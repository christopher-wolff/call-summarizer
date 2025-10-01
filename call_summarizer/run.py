#!/usr/bin/env python3
"""Combined script to extract audio from videos and transcribe them."""

from dataclasses import dataclass
from pathlib import Path

import dotenv

from call_summarizer import audio_extraction
from call_summarizer import audio_transcription
from call_summarizer import summaries


_SUMMARIZATION_PROMPT = """Please provide a comprehensive summary of the following conversation transcript.

Key points to include in the summary:
- Main topics discussed
- Key decisions made
- Action items or next steps
- Important details or agreements
- Overall tone and context

The majority of the summary should be focused on discussions around pricing, if they occur. If they do not discuss pricing of any products, please indicate.

The pricing conversation should take the following output:
- Pricing offer (ie. what was the amount of money estimated for the Omni products)
- Reaction to pricing (what was the prospect's feedback on the pricing, if any)
- Customer of budget, if indicated
- Pricing comparison to competitors, if any (did the customer tell us what competitor pricing is, and how they felt about). Please specify which competitors, if discussed.

Transcript:
{transcript_text}

Please provide a clear, structured summary that captures the essence of this conversation."""


@dataclass
class _Config:
    """Configuration for the call summarizer workflow."""
    
    # Directory configuration
    videos_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    summaries_dir: Path
    
    # Model configuration
    transcription_model: str
    summarization_model: str
    
    # Prompt configuration
    summarization_prompt: str


def _extract_audio_from_videos(config: _Config):
    """Extract audio from all video files in data/videos to data/audio."""
    input_dir = config.videos_dir
    output_dir = config.audio_dir
    
    # Ensure input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video file
    successful = 0
    skipped = 0
    
    for video_file in video_files:
        print(f"Processing: {video_file.name}")
        
        # Create output filename with .wav extension
        audio_filename = video_file.stem + ".wav"
        audio_path = output_dir / audio_filename
        
        # Skip if audio already exists
        if audio_path.exists():
            print(f"  ⏭ Skipping (audio already exists)")
            skipped += 1
            continue
        
        audio_extraction.extract_audio_from_video(str(video_file), str(audio_path))
        print(f"  ✓ Extracted to: {audio_path}")
        successful += 1
    
    print(f"\nAudio Extraction Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(video_files)}")


def _transcribe_audio_files(config: _Config):
    """Transcribe all audio files in data/audio to data/transcripts."""
    input_dir = config.audio_dir
    output_dir = config.transcripts_dir
    
    # Ensure input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported audio extensions
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    successful = 0
    skipped = 0
    
    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        
        # Create output filename with .json extension
        transcript_filename = audio_file.stem + ".json"
        transcript_path = output_dir / transcript_filename
        
        # Skip if transcript already exists
        if transcript_path.exists():
            print(f"  ⏭ Skipping (transcript already exists)")
            skipped += 1
            continue
        
        audio_transcription.transcribe_audio_file(str(audio_file), str(transcript_path), config.transcription_model)
        print(f"  ✓ Transcribed to: {transcript_path}")
        successful += 1
    
    print(f"\nTranscription Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(audio_files)}")


def _summarize_transcripts(config: _Config):
    """Summarize all transcript files in data/transcripts to data/summaries."""
    input_dir = config.transcripts_dir
    output_dir = config.summaries_dir
    
    # Ensure input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all transcript files
    transcript_files = list(input_dir.glob("*.json"))
    
    if not transcript_files:
        print(f"No transcript files found in {input_dir}")
        return
    
    print(f"Found {len(transcript_files)} transcript files")
    
    # Process each transcript file
    successful = 0
    skipped = 0
    
    for transcript_file in transcript_files:
        print(f"Processing: {transcript_file.name}")
        
        # Create output filename with .txt extension
        summary_filename = transcript_file.stem + "_summary.txt"
        summary_path = output_dir / summary_filename
        
        # Skip if summary already exists
        if summary_path.exists():
            print(f"  ⏭ Skipping (summary already exists)")
            skipped += 1
            continue
        
        # Temporarily update the prompt template
        original_prompt = summaries._PROMPT_TEMPLATE
        summaries._PROMPT_TEMPLATE = config.summarization_prompt
        
        summaries.summarize_transcript_file(str(transcript_file), str(summary_path), config.summarization_model)
        
        # Restore original prompt
        summaries._PROMPT_TEMPLATE = original_prompt
        print(f"  ✓ Summarized to: {summary_path}")
        successful += 1
    
    print(f"\nSummarization Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(transcript_files)}")


def _main(config: _Config):
    """Run the complete workflow: extract audio from videos, transcribe, and summarize."""
    print("=== Call Summarizer Workflow ===\n")
    
    # Step 1: Extract audio from videos
    print("Step 1: Extracting audio from videos...")
    _extract_audio_from_videos(config)
    
    print("\n" + "="*50 + "\n")
    
    # Step 2: Transcribe audio files
    print("Step 2: Transcribing audio files...")
    _transcribe_audio_files(config)
    
    print("\n" + "="*50 + "\n")
    
    # Step 3: Summarize transcripts
    print("Step 3: Summarizing transcripts...")
    _summarize_transcripts(config)
    
    print("\n" + "="*50 + "\n")
    print("✅ Workflow completed successfully!")
    print("\nResults:")
    print(f"  📁 Audio files: {config.audio_dir}")
    print(f"  📄 Transcripts: {config.transcripts_dir}")
    print(f"  📝 Summaries: {config.summaries_dir}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    
    # Create example configuration
    config = _Config(
        videos_dir=Path("data/videos"),
        audio_dir=Path("data/audio"),
        transcripts_dir=Path("data/transcripts"),
        summaries_dir=Path("data/summaries"),
        transcription_model="whisper-1",
        summarization_model="gpt-3.5-turbo",
        summarization_prompt=_SUMMARIZATION_PROMPT
    )
    
    _main(config)
