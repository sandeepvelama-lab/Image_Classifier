# Audio to Text UI — OpenAI Python SDK

A simple Streamlit web application that accepts an audio file or microphone recording and converts it to text using the OpenAI Python SDK.

## Features

- Upload MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM, OGG, or FLAC
- Record audio directly from the browser
- Choose a transcription model
- Optional language hint
- Display the transcript in the UI
- Download the transcript as a `.txt` file
- API key can be supplied through the sidebar or `OPENAI_API_KEY`

## 1. Install Python

Use Python 3.10 or newer.

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can use Command Prompt instead:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Set your API key

Option A — enter the key directly in the application's sidebar.

Option B — set an environment variable.

PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

Command Prompt:

```cmd
set OPENAI_API_KEY=YOUR_API_KEY
```

Do not commit your API key to Git.

## 5. Start the application

```powershell
streamlit run app.py
```

Then open the URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## How it works

The application calls:

```python
client.audio.transcriptions.create(
    model="gpt-4o-mini-transcribe",
    file=audio_file,
)
```

The returned transcription is available as:

```python
transcription.text
```

## Suggested project structure

```text
audio_to_text_app/
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

## Production considerations

For a production deployment, do not ask users to paste API keys into the browser UI. Store the API key securely on the server using environment variables or a secrets manager.

You can also add authentication, usage limits, logging, speaker diarization, SRT/VTT export, and a database for transcript history.
