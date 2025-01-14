import torch
from transformers import pipeline
import simple_llm as llm


# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Initialize the speech-to-text pipeline from Hugging Face Transformers
    # This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
    # The `chunk_length_s` parameter specifies the chunk length in seconds for processing
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

    # Perform speech recognition on the audio file
    # The `batch_size=8` parameter indicates how many chunks are processed at a time
    # The result is stored in `prediction` with the key "text" containing the transcribed text
    result = pipe(audio_file, batch_size=8)["text"]
    return llm.summarize_text(result)