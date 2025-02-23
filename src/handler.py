import runpod
import whisperx
import time

asr_options = { "task": "translate" }

model = whisperx.load_model(
    "large-v3", "cuda", asr_options=asr_options
)

def run_whisperx_job(job):
    start_time = time.time()

    job_input = job['input']
    url = job_input.get('url', "")

    print(f"🚧 Loading audio from {url}...")
    audio = whisperx.load_audio(url)
    print("✅ Audio loaded")

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=16)

    end_time = time.time()
    time_s = (end_time - start_time)
    print(f"🎉 Transcription done: {time_s:.2f} s")
    #print(result)

    # For easy migration, we are following the output format of runpod's 
    # official faster whisper.
    # https://github.com/runpod-workers/worker-faster_whisper/blob/main/src/predict.py#L111
    output = {
        'detected_language' : result['language'],
        'segments' : result['segments']
    }

    return output

runpod.serverless.start({"handler": run_whisperx_job})
