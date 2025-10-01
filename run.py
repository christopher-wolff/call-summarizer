#!/usr/bin/env python3
"""Combined script to extract audio from videos and transcribe them."""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the call_summarizer package to the path
sys.path.insert(0, str(Path(__file__).parent))

from call_summarizer.audio_extraction import extract_audio_from_video
from call_summarizer.audio_transcription import transcribe_audio_file


def extract_audio_from_videos():
    """Extract audio from all video files in data/videos to data/audio."""
    input_dir = Path("data/videos")
    output_dir = Path("data/audio")
    
    # Ensure input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return False
    
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
        return True
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video file
    successful = 0
    skipped = 0
    failed = 0
    
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
        
        success = extract_audio_from_video(str(video_file), str(audio_path))
        
        if success:
            print(f"  ✓ Extracted to: {audio_path}")
            successful += 1
        else:
            print(f"  ✗ Failed to extract audio")
            failed += 1
    
    print(f"\nAudio Extraction Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(video_files)}")
    
    return failed == 0


def transcribe_audio_files():
    """Transcribe all audio files in data/audio to data/transcripts."""
    input_dir = Path("data/audio")
    output_dir = Path("data/transcripts")
    
    # Ensure input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return False
    
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
        return True
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    successful = 0
    skipped = 0
    failed = 0
    
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
        
        success = transcribe_audio_file(str(audio_file), str(transcript_path))
        
        if success:
            print(f"  ✓ Transcribed to: {transcript_path}")
            successful += 1
        else:
            print(f"  ✗ Failed to transcribe audio")
            failed += 1
    
    print(f"\nTranscription Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(audio_files)}")
    
    return failed == 0


def main():
    """Run the complete workflow: extract audio from videos, then transcribe."""
    print("=== Call Summarizer Workflow ===\n")
    
    # Step 1: Extract audio from videos
    print("Step 1: Extracting audio from videos...")
    audio_success = extract_audio_from_videos()
    
    if not audio_success:
        print("\n❌ Audio extraction failed. Stopping workflow.")
        return 1
    
    print("\n" + "="*50 + "\n")
    
    # Step 2: Transcribe audio files
    print("Step 2: Transcribing audio files...")
    transcription_success = transcribe_audio_files()
    
    if not transcription_success:
        print("\n❌ Transcription failed.")
        return 1
    
    print("\n" + "="*50 + "\n")
    print("✅ Workflow completed successfully!")
    print("\nResults:")
    print("  📁 Audio files: data/audio/")
    print("  📄 Transcripts: data/transcripts/")
    
    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
