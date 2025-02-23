import runpod
import whisperx
import time

device="cuda"
model = whisperx.load_model("large-v3", device)

def run_whisperx_job(job):
    start_time = time.time()

    job_input = job['input']
    url = job_input.get('url', "")
    task = job_input.get('task', "transcribe")
    diarize = job_input.get('diarize', False)
    hf_token = job_input.get('hf_token', '')

    print(f"🚧 Loading audio from {url}...")
    audio = whisperx.load_audio(url)
    print("✅ Audio loaded")

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=16, task=task)

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

    if diarize:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        diarized_result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarized_result)
        del diarize_model

    return output

runpod.serverless.start({"handler": run_whisperx_job})
