# Call summarizer

This is a simple tool that automatically summarizes video calls.

1. Extract the audio from the videos as .wav files.
2. Generate transcripts using OpenAI's Whisper model.
3. Summarize the transcripts using one of OpenAI's language models.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/christopher-wolff/call-summarizer.git
   cd call-summarizer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Install FFmpeg:**
   ```
   brew install ffmpeg
   ```

5. **Set up your OpenAI API key:**
   ```bash
   cp env_template.txt .env
   ```
   Then, edit `.env` and add your OpenAI API key.

### Usage

1. **Add your video files** to the `data/videos/` directory

2. **Run the complete workflow:**
   ```bash
   python -m call_summarizer
   ```

   This will:
   - Extract audio from videos → `data/audio/`
   - Transcribe audio files → `data/transcripts/`
   - Generate summaries → `data/summaries/`

3. **Results:**
   - **Transcripts:** JSON files with full transcription data
   - **Summaries:** Text files with AI-generated summaries focused on pricing discussions

### Configuration

Edit the configuration in `call_summarizer/run.py` to customize:
- Model settings (Whisper, GPT models)
- Summarization prompt
- Directory paths
