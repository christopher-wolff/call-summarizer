"""Summarization utilities for generating summaries from transcripts."""

import os
from pathlib import Path
from typing import Optional

import openai

from call_summarizer import types

def summarize_transcript(transcript: types.Transcript, prompt_template: str, model: str = "gpt-3.5-turbo") -> str:
    """Generate a summary of a transcript using ChatGPT.
    
    Args:
        transcript: The transcript to summarize
        prompt_template: The prompt template to use for summarization (should contain {transcript_text})
        model: OpenAI model to use for summarization (default: gpt-3.5-turbo)
        
    Returns:
        Generated summary text
        
    Raises:
        ValueError: If OpenAI API key is not set
        openai.OpenAIError: If API call fails
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Create a prompt for summarization
    prompt = prompt_template.format(transcript_text=transcript.text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates clear, comprehensive summaries of business conversations and meetings."},
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content


def summarize_transcript_file(transcript_path: str, output_path: str, prompt_template: str, model: str = "gpt-3.5-turbo") -> bool:
    """Summarize a transcript file and save the result.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_path: Path where the summary should be saved
        prompt_template: The prompt template to use for summarization (should contain {transcript_text})
        model: OpenAI model to use for summarization
        
    Returns:
        True if summarization was successful, False otherwise
        
    Raises:
        FileNotFoundError: If the transcript file doesn't exist
        ValueError: If the transcript path is invalid
    """
    transcript_path_obj = Path(transcript_path)
    output_path_obj = Path(output_path)
    
    if not transcript_path_obj.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    if not transcript_path_obj.is_file():
        raise ValueError(f"Path is not a file: {transcript_path}")
    
    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the transcript
    transcript = types.Transcript.model_validate_json(transcript_path_obj.read_text(encoding='utf-8'))
    
    # Generate summary
    summary_text = summarize_transcript(transcript, prompt_template, model)
    
    # Save summary to file
    output_path_obj.write_text(summary_text, encoding='utf-8')
    
    return True


def get_summary_text(summary_path: str) -> str:
    """Get the summary text from a summary file.
    
    Args:
        summary_path: Path to the summary text file
        
    Returns:
        The summary text
        
    Raises:
        FileNotFoundError: If the summary file doesn't exist
    """
    summary_path_obj = Path(summary_path)
    
    if not summary_path_obj.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    return summary_path_obj.read_text(encoding='utf-8')
