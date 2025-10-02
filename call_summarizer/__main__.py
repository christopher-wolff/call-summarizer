#!/usr/bin/env python3
"""Combined script to extract audio from videos and transcribe them."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

import dotenv

from call_summarizer import audio_extraction
from call_summarizer import audio_transcription
from call_summarizer import summaries
from call_summarizer.progress import PipelineProgress


# Private dataclasses for internal use only
@dataclass
class _ProcessingResult:
    """Result of processing a single file."""
    success: bool
    message: str
    skipped: bool = False


@dataclass
class _ChunkTranscriptionResult:
    """Result of transcribing a single audio chunk."""
    success: bool
    transcript_data: dict | None
    message: str


@dataclass
class _StageSummary:
    """Summary of processing results for a pipeline stage."""
    successful: int
    skipped: int
    failed: int
    total: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total == 0:
            return 0.0
        return (self.successful / self.total) * 100


def _retry_rpc_call(func, *args, max_retries=3, **kwargs):
    """Retry an RPC call up to max_retries times, then fail loudly."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            break
    
    raise RuntimeError(f"RPC call failed after {max_retries} attempts: {last_exception}")


def _create_output_if_needed(file_path: Path) -> Path:
    """Ensure output directory exists and return the path."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _log_and_fail(message: str):
    """Log error message and fail loudly."""
    print(f"❌ FATAL ERROR: {message}")
    raise RuntimeError(message)


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
        return [audio_file]
    
    # Calculate chunks needed and duration
    num_chunks = (file_size + max_size_bytes - 1) // max_size_bytes
    total_duration = _get_audio_duration(audio_file)
    chunk_duration = total_duration / num_chunks
    
    # Create chunks in parallel
    filename_hash = hashlib.md5(str(audio_file).encode()).hexdigest()[:8]
    unique_prefix = f"chunk_{audio_file.stat().st_ino}_{filename_hash}"
    
    chunks = []
    for i in range(num_chunks):
        chunk_path = _create_chunk(audio_file, temp_dir, unique_prefix, i, chunk_duration)
        chunks.append(chunk_path)
    
    return chunks


def _get_audio_duration(audio_file: Path) -> float:
    """Get audio duration using ffprobe."""
    duration_cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', str(audio_file)
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _create_chunk(audio_file: Path, temp_dir: Path, prefix: str, chunk_index: int, chunk_duration: float) -> Path:
    """Create a single audio chunk."""
    start_time = chunk_index * chunk_duration
    chunk_filename = f"{prefix}_{chunk_index:03d}.wav"
    chunk_path = temp_dir / chunk_filename
    
    split_cmd = [
        'ffmpeg', '-i', str(audio_file),
        '-ss', str(start_time),
        '-t', str(chunk_duration),
        '-ac', '1', '-ar', '16000', '-b:a', '64k',
        '-avoid_negative_ts', 'make_zero',
        '-y', str(chunk_path)
    ]
    
    subprocess.run(split_cmd, capture_output=True, text=True, check=True)
    
    if not chunk_path.exists():
        _log_and_fail(f"Chunk file was not created: {chunk_path}")
    
    return chunk_path


def _merge_transcripts(transcripts: list, original_filename: str) -> dict | None:
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


def _process_video_file(video_file: Path, audio_dir: Path) -> _ProcessingResult:
    """Process a single video file to extract and compress audio."""
    audio_filename = video_file.stem + ".wav"
    audio_path = audio_dir / audio_filename
    
    if audio_path.exists():
        return _ProcessingResult(
            success=True, 
            message=f"⏭ Skipped {video_file.name} (audio already exists)",
            skipped=True
        )
    
    # Extract audio with compression for Whisper API
    cmd = [
        'ffmpeg', '-i', str(video_file),
        '-ac', '1',  # Mono
        '-ar', '16000',  # 16kHz
        '-b:a', '64k',  # 64kbps
        '-y', str(audio_path)
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Verify extraction and return result
    final_size = audio_path.stat().st_size
    size_mb = final_size / (1024 * 1024)
    return _ProcessingResult(
        success=True,
        message=f"✅ Extracted audio ({size_mb:.1f}MB): {audio_path}"
    )


def _transcribe_chunk(chunk_path: Path, chunk_transcript_path: Path, model: str, chunk_index: int, audio_file_name: str, total_chunks: int, progress: PipelineProgress) -> _ChunkTranscriptionResult:
    """Transcribe a single audio chunk with retry logic."""
    progress.update_transcription_chunk(audio_file_name, chunk_index, total_chunks)
    
    # Transcribe with retry logic for API calls
    _retry_rpc_call(
        audio_transcription.transcribe_audio_file,
        str(chunk_path),
        str(chunk_transcript_path), 
        model
    )
    
    # Load and process transcript
    with open(chunk_transcript_path, 'r', encoding='utf-8') as f:
        chunk_transcript = json.load(f)
    chunk_transcript['chunk_order'] = chunk_index
    
    # Clean up chunk transcript file
    chunk_transcript_path.unlink()
    
    return _ChunkTranscriptionResult(
        success=True,
        transcript_data=chunk_transcript,
        message=f"✅ Transcribed chunk {chunk_index + 1}"
    )


def _process_audio_file(audio_file: Path, transcripts_dir: Path, model: str, progress: PipelineProgress, file_id: int = 0) -> _ProcessingResult:
    """Process a single audio file to create transcript, with automatic chunking for large files."""
    transcript_filename = audio_file.stem + ".json"
    transcript_path = transcripts_dir / transcript_filename
    
    if transcript_path.exists():
        return _ProcessingResult(
            success=True,
            message=f"⏭ Skipped (transcript already exists)",
            skipped=True
        )
    
    # Check file size (OpenAI Whisper limit is 25MB)
    file_size = audio_file.stat().st_size
    max_size = 25 * 1024 * 1024  # 25MB in bytes
    
    if file_size <= max_size:
        _retry_rpc_call(audio_transcription.transcribe_audio_file, str(audio_file), str(transcript_path), model)
        return _ProcessingResult(
            success=True,
            message=f"✅ Transcribed: {transcript_path}"
        )
    
    # File is too large, need to split it
    return _process_large_audio_file(audio_file, transcript_path, transcripts_dir, model, progress, file_size, max_size)


def _process_large_audio_file(audio_file: Path, transcript_path: Path, transcripts_dir: Path, model: str, progress: PipelineProgress, file_size: int, max_size: int) -> _ProcessingResult:
    """Process large audio files with chunking."""
    temp_dir = transcripts_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        chunks = _split_audio_file(audio_file, temp_dir, max_size)
        
        if len(chunks) > 1:
            progress.update_transcription_task(audio_file.name, 0, total=len(chunks))
        
        if len(chunks) == 1:
            _retry_rpc_call(audio_transcription.transcribe_audio_file, str(audio_file), str(transcript_path), model)
            return _ProcessingResult(
                success=True,
                message=f"✅ Transcribed (direct): {transcript_path}"
            )
        
        chunk_transcripts = _transcribe_chunks_parallel(chunks, transcript_path, model, audio_file.name, progress)
        merged_transcript = _merge_transcripts(chunk_transcripts, audio_file.name)
        
        if not merged_transcript:
            _log_and_fail(f"Failed to merge transcripts for {audio_file.name}")
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(merged_transcript, f, indent=2, ensure_ascii=False)
        
        return _ProcessingResult(
            success=True,
            message=f"✅ Transcribed ({len(chunks)} chunks, {file_size/(1024*1024):.1f}MB): {transcript_path}"
        )
    
    finally:
        _cleanup_temp_chunks(temp_dir)


def _transcribe_chunks_parallel(chunks: list[Path], transcript_path: Path, model: str, audio_file_name: str, progress: PipelineProgress) -> list[dict]:
    """Transcribe chunks in parallel."""
    chunk_transcripts = []
    
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_chunk = {}
        for i, chunk_path in enumerate(chunks):
            chunk_transcript_path = transcript_path.parent / f"{transcript_path.stem}_chunk_{i:03d}.json"
            future = executor.submit(
                _transcribe_chunk, 
                chunk_path, 
                chunk_transcript_path, 
                model, 
                i, 
                audio_file_name, 
                len(chunks),
                progress
            )
            future_to_chunk[future] = i
        
        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            result = future.result()
            
            if not result.success or not result.transcript_data:
                _log_and_fail(f"Transcription failed for chunk {chunk_index}: {result.message}")
            
            chunk_transcripts.append(result.transcript_data)
    
    return chunk_transcripts


def _cleanup_temp_chunks(temp_dir: Path):
    """Clean up temporary chunk files."""
    if not temp_dir.exists():
        return
        
    for chunk_file in temp_dir.glob("*.wav"):
        chunk_file.unlink()
        
    if not any(temp_dir.iterdir()):
        temp_dir.rmdir()


def _count_video_files(videos_dir: Path, limit: int | None = None) -> list[Path]:
    """Count video files in the videos directory."""
    if not videos_dir.exists():
        return []
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(f"*{ext}"))
        video_files.extend(videos_dir.glob(f"*{ext.upper()}"))
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        video_files = video_files[:limit]
    
    return video_files


def _count_audio_files(audio_dir: Path, limit: int | None = None) -> list[Path]:
    """Count audio files in the audio directory."""
    if not audio_dir.exists():
        return []
    
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        audio_files = audio_files[:limit]
    
    return audio_files


def _count_transcript_files(transcripts_dir: Path, limit: int | None = None) -> list[Path]:
    """Count transcript files in the transcripts directory."""
    if not transcripts_dir.exists():
        return []
    
    transcript_files = list(transcripts_dir.glob("*.json"))
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        transcript_files = transcript_files[:limit]
    
    return transcript_files


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
    
    # Processing configuration
    limit: int | None = None  # Limit number of files to process (for testing)


def _process_transcript_file(transcript_file: Path, summaries_dir: Path, model: str, prompt: str) -> _ProcessingResult:
    """Process a single transcript file to create summary."""
    summary_filename = transcript_file.stem + ".txt"
    summary_path = summaries_dir / summary_filename
    
    if summary_path.exists():
        return _ProcessingResult(
            success=True,
            message=f"⏭ Skipped {transcript_file.name} (summary already exists)",
            skipped=True
        )
    
    _retry_rpc_call(summaries.summarize_transcript_file, str(transcript_file), str(summary_path), prompt, model)
    return _ProcessingResult(
        success=True,
        message=f"✅ Summarized: {summary_path}"
    )


def _extract_audio_from_videos(config: _Config, progress: PipelineProgress) -> _StageSummary:
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
    
    # Apply limit if specified
    if config.limit is not None and config.limit > 0:
        video_files = video_files[:config.limit]
    
    if not video_files:
        return _StageSummary(successful=0, skipped=0, failed=0, total=0)
    
    # Progress tracking is set up in main function
    
    # Add all tasks to progress tracker
    for video_file in video_files:
        progress.add_audio_task(video_file.name)
    
    # Process files in parallel
    successful = 0
    skipped = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=128) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_video_file, video_file, output_dir): video_file
            for video_file in video_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            video_file = future_to_file[future]
            result = future.result()
            
            if result.skipped:
                skipped += 1
                progress.complete_audio_task(video_file.name, success=True)
            elif not result.success:
                failed += 1
                progress.complete_audio_task(video_file.name, success=False)
            else:
                successful += 1
                progress.complete_audio_task(video_file.name, success=True)
    
    return _StageSummary(successful=successful, skipped=skipped, failed=failed, total=len(video_files))


def _transcribe_audio_files(config: _Config, progress: PipelineProgress) -> _StageSummary:
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
    
    # Apply limit if specified
    if config.limit is not None and config.limit > 0:
        audio_files = audio_files[:config.limit]
    
    if not audio_files:
        return _StageSummary(successful=0, skipped=0, failed=0, total=0)
    
    # Progress tracking is set up in main function
    # Add all tasks to progress tracker (will be updated with actual chunk count if needed)
    for audio_file in audio_files:
        progress.add_transcription_task(audio_file.name, total_chunks=1)
    
    # Process files in parallel with higher concurrency for transcription
    successful = 0
    skipped = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=128) as executor:  # High concurrency for transcription
        # Submit all tasks with unique IDs
        future_to_info = {}
        for i, audio_file in enumerate(audio_files):
            future = executor.submit(_process_audio_file, audio_file, output_dir, config.transcription_model, progress, i+1)
            future_to_info[future] = (audio_file, i+1)
        
        # Process completed tasks
        for future in as_completed(future_to_info):
            audio_file, file_id = future_to_info[future]
            result = future.result()
            
            if result.skipped:
                skipped += 1
                progress.complete_transcription_task(audio_file.name, success=True)
            elif not result.success:
                failed += 1
                progress.complete_transcription_task(audio_file.name, success=False)
            else:
                successful += 1
                progress.complete_transcription_task(audio_file.name, success=True)
    
    return _StageSummary(successful=successful, skipped=skipped, failed=failed, total=len(audio_files))


def _summarize_transcripts(config: _Config, progress: PipelineProgress) -> _StageSummary:
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
    
    # Apply limit if specified
    if config.limit is not None and config.limit > 0:
        transcript_files = transcript_files[:config.limit]
    
    if not transcript_files:
        return _StageSummary(successful=0, skipped=0, failed=0, total=0)
    
    # Progress tracking is set up in main function
    
    # Add all tasks to progress tracker
    for transcript_file in transcript_files:
        progress.add_summarization_task(transcript_file.name)
    
    # Process files in parallel
    successful = 0
    skipped = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=128) as executor:  # High concurrency for API calls
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_transcript_file, transcript_file, output_dir, config.summarization_model, config.summarization_prompt): transcript_file
            for transcript_file in transcript_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            transcript_file = future_to_file[future]
            result = future.result()
            
            if result.skipped:
                skipped += 1
                progress.complete_summarization_task(transcript_file.name, success=True)
            elif not result.success:
                failed += 1
                progress.complete_summarization_task(transcript_file.name, success=False)
            else:
                successful += 1
                progress.complete_summarization_task(transcript_file.name, success=True)
    
    return _StageSummary(successful=successful, skipped=skipped, failed=failed, total=len(transcript_files))


def _main(config: _Config):
    """Run the complete workflow: extract audio from videos, transcribe, and summarize."""
    progress = PipelineProgress()
    
    try:
        progress.start()
        
        # Count files for each stage to set up progress tracking
        video_files = _count_video_files(config.videos_dir, config.limit)
        
        # Set up progress stages upfront
        progress.setup_audio_stage(len(video_files))
        
        # Step 1: Extract audio from videos
        audio_results = _extract_audio_from_videos(config, progress)
        
        # Step 2: Set up transcription stage after audio extraction
        # Count the actual audio files that were created
        actual_audio_files = _count_audio_files(config.audio_dir, config.limit)
        progress.update_transcription_stage(len(actual_audio_files))
        
        # Step 3: Transcribe audio files
        transcription_results = _transcribe_audio_files(config, progress)
        
        # Step 4: Set up summarization stage after transcription
        # Count the actual transcript files that were created
        actual_transcript_files = _count_transcript_files(config.transcripts_dir, config.limit)
        progress.update_summarization_stage(len(actual_transcript_files))
        
        # Step 5: Summarize transcripts
        summarization_results = _summarize_transcripts(config, progress)
        
        # Show final summary
        results = {
            "audio": audio_results,
            "transcription": transcription_results,
            "summarization": summarization_results
        }
        progress.show_final_summary(results)
        
    finally:
        progress.stop()
        # Give Rich time to clean up the display
        time.sleep(0.2)
    
    # Clear screen and show clean final results
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("✅ Workflow completed successfully!")
    print("\nResults:")
    print(f"  📁 Audio files: {config.audio_dir}")
    print(f"  📄 Transcripts: {config.transcripts_dir}")
    print(f"  📝 Summaries: {config.summaries_dir}")
    
    # Show final summary table again in clean format
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Audio Extraction:     {audio_results.successful} successful, {audio_results.failed} failed, {audio_results.skipped} skipped")
    print(f"Transcription:        {transcription_results.successful} successful, {transcription_results.failed} failed, {transcription_results.skipped} skipped")
    print(f"Summarization:        {summarization_results.successful} successful, {summarization_results.failed} failed, {summarization_results.skipped} skipped")
    print(f"Overall Success Rate: {((audio_results.successful + transcription_results.successful + summarization_results.successful) / (audio_results.total + transcription_results.total + summarization_results.total) * 100):.1f}%")
    print("="*60)


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
        summarization_prompt=_SUMMARIZATION_PROMPT,
        limit=None,
    )
    
    _main(config)