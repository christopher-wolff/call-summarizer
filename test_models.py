#!/usr/bin/env python3
"""Test script to demonstrate Pydantic models for transcript files."""

from call_summarizer.models import Transcript, TranscriptSegment
from call_summarizer.audio_transcription import load_transcript

def test_models():
    """Test the Pydantic models with the existing transcript file."""
    
    # Test creating a Transcript model manually
    print("=== Testing Pydantic Models ===\n")
    
    # Create a sample segment
    segment = TranscriptSegment(
        id=0,
        start=0.0,
        end=8.0,
        text="Hello, this is a test segment."
    )
    
    print("✅ Created TranscriptSegment:")
    print(f"   ID: {segment.id}")
    print(f"   Time: {segment.start}s - {segment.end}s")
    print(f"   Text: {segment.text}")
    print()
    
    # Create a full transcript
    transcript = Transcript(
        text="Hello, this is a test transcript.",
        language="english",
        duration=8.0,
        segments=[segment]
    )
    
    print("✅ Created Transcript:")
    print(f"   Language: {transcript.language}")
    print(f"   Duration: {transcript.duration}s")
    print(f"   Segments: {len(transcript.segments)}")
    print()
    
    # Test JSON serialization
    json_data = transcript.model_dump()
    print("✅ JSON serialization works:")
    print(f"   Keys: {list(json_data.keys())}")
    print()
    
    # Test loading existing transcript file
    try:
        existing_transcript = load_transcript("data/transcripts/example_call.json")
        print("✅ Loaded existing transcript file:")
        print(f"   Language: {existing_transcript.language}")
        print(f"   Duration: {existing_transcript.duration:.1f}s")
        print(f"   Segments: {len(existing_transcript.segments)}")
        print(f"   Text preview: {existing_transcript.text[:100]}...")
        
    except FileNotFoundError:
        print("⚠️  No existing transcript file found to test with")
    except Exception as e:
        print(f"❌ Error loading transcript: {e}")
    
    print("\n=== Model Validation ===")
    
    # Test validation
    try:
        # This should work
        valid_segment = TranscriptSegment(
            id=1,
            start=1.0,
            end=2.0,
            text="Valid segment"
        )
        print("✅ Valid segment created successfully")
        
        # This should fail validation
        try:
            invalid_segment = TranscriptSegment(
                id="invalid",  # Should be int
                start=1.0,
                end=2.0,
                text="Invalid segment"
            )
            print("❌ Invalid segment was accepted (this shouldn't happen)")
        except Exception as e:
            print(f"✅ Invalid segment correctly rejected: {type(e).__name__}")
            
    except Exception as e:
        print(f"❌ Error testing validation: {e}")

if __name__ == "__main__":
    test_models()
