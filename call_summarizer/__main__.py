#!/usr/bin/env python3
"""Combined script to extract audio from videos and transcribe them."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import subprocess

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
- Pricing offer (ie. what was the amount of money estimated for the Omni products), with specifics in terms of how much was offered for the platform vs. user licenses
- Reaction to pricing (what was the prospect's feedback on the pricing, if any)
- Customer of budget, if indicated
- Pricing comparison to competitors, if any (did the customer tell us what competitor pricing is, and how they felt about). Please specify which competitors, if discussed.

Transcript:
{transcript_text}

Please provide a clear, structured summary that captures the essence of this conversation."""


def _split_audio_file(audio_file: Path, temp_dir: Path, max_size_bytes: int) -> list[Path]:
    """Split a large audio file into chunks that fit within the size limit."""
    file_size = audio_file.stat().st_size
    
    if file_size <= max_size_bytes:
        return [audio_file]  # No splitting needed
    
    # Calculate number of chunks needed
    num_chunks = (file_size + max_size_bytes - 1) // max_size_bytes
    
    # Get audio duration to calculate chunk duration
    duration_cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', str(audio_file)
    ]
    
    result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
    total_duration = float(result.stdout.strip())
    chunk_duration = total_duration / num_chunks
    
    # Create chunks
    chunks = []
    # Use a unique prefix based on the file's inode and a hash of the filename
    filename_hash = hashlib.md5(str(audio_file).encode()).hexdigest()[:8]
    unique_prefix = f"chunk_{audio_file.stat().st_ino}_{filename_hash}"
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_filename = f"{unique_prefix}_{i:03d}.wav"
        chunk_path = temp_dir / chunk_filename
        
        split_cmd = [
            'ffmpeg', '-i', str(audio_file),
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-ac', '1', '-ar', '16000', '-b:a', '64k',
            '-avoid_negative_ts', 'make_zero',
            '-y', str(chunk_path)
        ]
        
        try:
            result = subprocess.run(split_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed for chunk {i}: {e.stderr}")
        
        # Verify the chunk file was actually created
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file was not created: {chunk_path}")
        
        chunks.append(chunk_path)
    
    return chunks


def _merge_transcripts(transcripts: list, original_filename: str) -> dict:
    """Merge multiple transcript chunks into a single seamless transcript."""
    if not transcripts:
        return None
    
    if len(transcripts) == 1:
        return transcripts[0]
    
    # Sort transcripts by their chunk order (extract from filename or order)
    sorted_transcripts = sorted(transcripts, key=lambda t: t.get('chunk_order', 0))
    
    # Merge the text
    merged_text = " ".join(t['text'] for t in sorted_transcripts if t['text'])
    
    # Merge segments with adjusted timestamps
    merged_segments = []
    current_time_offset = 0.0
    
    for transcript in sorted_transcripts:
        if 'segments' in transcript and transcript['segments']:
            for segment in transcript['segments']:
                # Adjust timestamps by adding the current time offset
                adjusted_segment = {
                    'id': len(merged_segments),
                    'start': segment['start'] + current_time_offset,
                    'end': segment['end'] + current_time_offset,
                    'text': segment['text']
                }
                merged_segments.append(adjusted_segment)
        
        # Update time offset for next chunk
        if transcript.get('duration'):
            current_time_offset += transcript['duration']
    
    # Use metadata from first transcript
    first_transcript = sorted_transcripts[0]
    merged_transcript = {
        'text': merged_text,
        'language': first_transcript.get('language', 'en'),
        'duration': sum(t.get('duration', 0) for t in sorted_transcripts),
        'segments': merged_segments
    }
    
    return merged_transcript


def _process_video_file(video_file: Path, audio_dir: Path) -> tuple[bool, str]:
    """Process a single video file to extract and compress audio."""
    audio_filename = video_file.stem + ".wav"
    audio_path = audio_dir / audio_filename
    
    if audio_path.exists():
        return True, f"⏭ Skipped {video_file.name} (audio already exists)"
    
    # Extract audio with compression settings optimized for Whisper API
    # Single channel (mono), 16kHz sample rate, compressed bitrate
    # Use ffmpeg directly for better compression control
    cmd = [
        'ffmpeg', '-i', str(video_file),
        '-ac', '1',  # Single channel (mono)
        '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
        '-b:a', '64k',  # 64kbps bitrate for good compression
        '-y',  # Overwrite output file
        str(audio_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check final file size
        final_size = audio_path.stat().st_size
        size_mb = final_size / (1024 * 1024)
        
        return True, f"✓ Extracted audio ({size_mb:.1f}MB): {audio_path}"
    except subprocess.CalledProcessError as e:
        return False, f"✗ FFmpeg error: {e.stderr.strip()}"
    except Exception as e:
        return False, f"✗ Error: {str(e)}"


def _process_audio_file(audio_file: Path, transcripts_dir: Path, model: str, file_id: int = 0) -> tuple[bool, str]:
    """Process a single audio file to create transcript, with automatic chunking for large files."""
    transcript_filename = audio_file.stem + ".json"
    transcript_path = transcripts_dir / transcript_filename
    
    if transcript_path.exists():
        return True, f"⏭ Skipped (transcript already exists)"
    
    # Check file size (OpenAI Whisper limit is 25MB)
    file_size = audio_file.stat().st_size
    max_size = 25 * 1024 * 1024  # 25MB in bytes
    
    if file_size <= max_size:
        # File is small enough, transcribe directly
        audio_transcription.transcribe_audio_file(str(audio_file), str(transcript_path), model)
        return True, f"✓ Transcribed: {transcript_path}"
    
    # File is too large, need to split it
    temp_dir = transcripts_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Split the audio file into chunks
        chunks = _split_audio_file(audio_file, temp_dir, max_size)
        
        if len(chunks) == 1:
            # Splitting didn't help, try direct transcription anyway
            audio_transcription.transcribe_audio_file(str(audio_file), str(transcript_path), model)
            return True, f"✓ Transcribed (direct): {transcript_path}"
        
        # Transcribe each chunk sequentially to avoid rate limiting
        chunk_transcripts = []
        
        for i, chunk_path in enumerate(chunks):
            chunk_transcript_path = transcript_path.parent / f"{transcript_path.stem}_chunk_{i:03d}.json"
            
            try:
                audio_transcription.transcribe_audio_file(str(chunk_path), str(chunk_transcript_path), model)
                
                # Load the chunk transcript
                with open(chunk_transcript_path, 'r', encoding='utf-8') as f:
                    chunk_transcript = json.load(f)
                chunk_transcript['chunk_order'] = i
                chunk_transcripts.append(chunk_transcript)
                
                # Clean up chunk transcript file
                chunk_transcript_path.unlink()
                
                print(f"[{file_id:02d}] ✓ Transcribed chunk {i+1}/{len(chunks)}")
                
            except Exception as e:
                return False, f"✗ Error transcribing chunk {i}: {str(e)}"
        
        # Merge all chunk transcripts
        merged_transcript = _merge_transcripts(chunk_transcripts, audio_file.name)
        
        if merged_transcript:
            # Save the merged transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(merged_transcript, f, indent=2, ensure_ascii=False)
            
            return True, f"✓ Transcribed ({len(chunks)} chunks, {file_size/(1024*1024):.1f}MB): {transcript_path}"
        else:
            return False, f"✗ Failed to merge transcripts for {audio_file.name}"
    
    finally:
        # Clean up temporary chunk files
        if temp_dir.exists():
            for chunk_file in temp_dir.glob("*.wav"):
                chunk_file.unlink()
            if not any(temp_dir.iterdir()):
                temp_dir.rmdir()


def _process_transcript_file(transcript_file: Path, summaries_dir: Path, model: str, prompt: str) -> tuple[bool, str]:
    """Process a single transcript file to create summary."""
    summary_filename = transcript_file.stem + ".txt"
    summary_path = summaries_dir / summary_filename
    
    if summary_path.exists():
        return True, f"⏭ Skipped {transcript_file.name} (summary already exists)"
    
    summaries.summarize_transcript_file(str(transcript_file), str(summary_path), prompt, model)
    return True, f"✓ Summarized: {summary_path}"


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
    
    # Process files in parallel
    successful = 0
    skipped = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_video_file, video_file, output_dir): video_file
            for video_file in video_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            video_file = future_to_file[future]
            try:
                success, message = future.result()
                print(f"Processing: {video_file.name}")
                print(f"  {message}")
                if "Skipped" in message:
                    skipped += 1
                elif "✗" in message or not success:
                    failed += 1
                else:
                    successful += 1
            except Exception as exc:
                print(f"Processing: {video_file.name}")
                print(f"  ✗ Error: {exc}")
                failed += 1
    
    print(f"\nAudio Extraction Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
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
    
    # Process files in parallel with higher concurrency for transcription
    successful = 0
    skipped = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:  # Higher concurrency for transcription
        # Submit all tasks with unique IDs
        future_to_info = {}
        for i, audio_file in enumerate(audio_files):
            future = executor.submit(_process_audio_file, audio_file, output_dir, config.transcription_model, i+1)
            future_to_info[future] = (audio_file, i+1)
        
        # Process completed tasks
        for future in as_completed(future_to_info):
            audio_file, file_id = future_to_info[future]
            try:
                success, message = future.result()
                print(f"[{file_id:02d}] Processing: {audio_file.name}")
                print(f"[{file_id:02d}]   {message}")
                if "Skipped" in message:
                    skipped += 1
                elif "✗" in message or not success:
                    failed += 1
                else:
                    successful += 1
            except Exception as exc:
                print(f"[{file_id:02d}] Processing: {audio_file.name}")
                print(f"[{file_id:02d}]   ✗ Error: {exc}")
                failed += 1
    
    print(f"\nTranscription Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
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
    
    # Process files in parallel
    successful = 0
    skipped = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=2) as executor:  # Lower concurrency for API calls
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_transcript_file, transcript_file, output_dir, config.summarization_model, config.summarization_prompt): transcript_file
            for transcript_file in transcript_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            transcript_file = future_to_file[future]
            try:
                success, message = future.result()
                print(f"Processing: {transcript_file.name}")
                print(f"  {message}")
                if "Skipped" in message:
                    skipped += 1
                elif "✗" in message or not success:
                    failed += 1
                else:
                    successful += 1
            except Exception as exc:
                print(f"Processing: {transcript_file.name}")
                print(f"  ✗ Error: {exc}")
                failed += 1
    
    print(f"\nSummarization Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
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
        summarization_model="gpt-5",
        summarization_prompt=_SUMMARIZATION_PROMPT
    )
    
    _main(config)