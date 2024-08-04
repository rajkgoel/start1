from transformers import pipeline

transcriber = pipeline(model="openai/whisper-large-v2")
audio_filenames = f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
audio_filenames = [
        f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)
    ]
texts = transcriber(audio_filenames)

for result in texts:
    print("--------------------------------------------")
    print(result["text"])
