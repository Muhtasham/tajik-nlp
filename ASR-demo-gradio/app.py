import whisper
import gradio as gr

model = whisper.load_model("medium")

def transcribe(audio):
    
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    
    tg_prob = probs["tg"]
    fa_prob = probs["fa"]
    
    print(f"Probality of your speech being Tajik language: {tg_prob}")
    print(f"Probality of your speech being Persian (Farsi) language: {fa_prob}")
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False,language="tg")
    result = whisper.decode(model, mel, options)
    return result.text
    
    
 
gr.Interface(
    title = 'OpenAI Whisper Tajik ASR Gradio Web UI', 
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()
