import wave
import numpy as np
import requests as requests
import torch

from pathlib import Path
from flask import Flask, request

from model import Model
from audioUtils import AudioUtil
from soundClass import SoundClass

response = requests.get('https://firewave-models.s3.eu-central-1.amazonaws.com/model.pt')
open(Path.cwd() / 'model.pt', 'wb').write(response.content)
model = Model.load_from_file(Path.cwd() / 'model.pt')

app = Flask(__name__)

@app.route('/')
def index():
    return 'It Works!!!'


@app.post('/upload')
def upload():
    file = request.files['file_data']
    with wave.open('tmp.wav', 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        waveform = file.stream.read()
        wavfile.writeframes(waveform)
    aud = AudioUtil.open('tmp.wav')
    reaud = AudioUtil.resample(aud, 16000)
    rechan = AudioUtil.rechannel(reaud, 2)
    dur_aud = AudioUtil.pad_trunc(rechan, 4000)

    spectro_gram = AudioUtil.spectro_gram(dur_aud)
    inputs = spectro_gram[np.newaxis, ...]
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
    with torch.no_grad():
        outputs = model(inputs)
    _, prediction = torch.max(outputs, 1)
    print(prediction)
    return SoundClass(prediction.item()).name

# if __name__ == "__main__":
#     app.run()
