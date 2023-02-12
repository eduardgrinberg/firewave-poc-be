import wave
import numpy as np
import torch

from pathlib import Path
from flask import Flask, request

from model import Model
from audioUtils import AudioUtil
from soundClass import SoundClass

model = Model.load_from_file(Path.cwd() / 'model.pt')

app = Flask(__name__)

@app.route('/')
def index():
    return 'It Works!!!'


@app.post('/upload')
def upload():
    file = request.files['file_data']
    with wave.open('test.wav', 'wb') as wavfile:
        wavfile.setparams((2, 2, 44100, 0, 'NONE', 'NONE'))
        waveform = file.stream.read()
        wavfile.writeframes(waveform)
    test_file_data = AudioUtil.open('test.wav')
    spectro_gram = AudioUtil.spectro_gram(test_file_data)
    inputs = spectro_gram[np.newaxis, ...]
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
    outputs = model(inputs)
    _, prediction = torch.max(outputs, 1)
    print(prediction)
    return SoundClass(prediction.item()).name

# if __name__ == "__main__":
#     app.run()
