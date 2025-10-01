# Call Summarizer

A tool for analyzing and summarizing sales calls from video files.

## Getting started


1. **Set up a virtual environment**

```sh
python -m venv .venv
source .venv/bin/activate
```

2. **Install required dependencies**

```sh
pip install -r requirements.txt
```

3. **Set up environment variables**

Copy the template and add your OpenAI API key:
```sh
cp env_template.txt .env
# Edit .env and add your actual OpenAI API key
```

Get your API key from: https://platform.openai.com/api-keys

4. **Install ffmpeg**

```sh
brew install ffmpeg
```

5. **Run the complete workflow**

Run the entire pipeline (extract audio + transcribe):
```sh
python run.py
```

The workflow is resumable - if you run it again, it will skip any steps where the output files already exist.
