import torch
import whisper
import os
import assemblyai as aai

# Create data/transcripts directory if it doesn't exist
if not os.path.exists('data/transcripts'):
    os.makedirs('data/transcripts')

def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using OpenAI Whisper
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
    
    Returns:
        str: The transcribed text
    """
    try:
        print(f"Transcribing {audio_file_path}...")
        # # Load the Whisper model
        # model = whisper.load_model("base") 

        aai.settings.api_key = os.getenv('AAI_API_KEY')
        transciber = aai.Transcriber()

        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = transciber.transcribe(audio_file_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription failed: {transcript.error}")
            exit(1)

        
        
        # Transcribe the audio file
        # result = model.transcribe(audio_file_path)
        
        # # Get the transcribed text
        # transcript = result["text"]
        
        # Save transcription with speaker labels
        output_file = os.path.join('data/transcripts', os.path.basename(audio_file_path).replace('.mp3', '.txt'))
        with open(output_file, "w", encoding="utf-8") as f:
            for utterance in transcript.utterances:
                f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")
        # # Save transcription to file
        # output_file = os.path.join('data/transcripts', os.path.basename(audio_file_path).replace('.mp3', '.txt'))
        # with open(output_file, "w", encoding="utf-8") as f:
        #     f.write(transcript.text)
            
        print(f"Transcription saved to {output_file}")
        return transcript
        
    except Exception as e:
        print(f"Error transcribing {audio_file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage - transcribe all audio files in downloads directory
    downloads_dir = "data/audio"
    if os.path.exists(downloads_dir):
        for filename in os.listdir(downloads_dir):
            if filename.endswith((".mp3", ".wav", ".m4a")):
                audio_path = os.path.join(downloads_dir, filename)
                transcribe_audio(audio_path)
