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
    align = job_input.get('align', False)
    diarize = job_input.get('diarize', False)
    hf_token = job_input.get('hf_token', '')
    min_speakers = job_input.get('min_speakers', None)
    max_speakers = job_input.get('max_speakers', None)

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

    if align:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    if diarize:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_model

    output = {
        'segments' : result['segments']
    }
    
    return output

runpod.serverless.start({"handler": run_whisperx_job})
