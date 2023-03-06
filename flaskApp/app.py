import os
import random
import wave
from time import time

import numpy as np
import requests as requests
import torch

from pathlib import Path
from flask import Flask, request
from torchsummary import summary

from model import Model
from audioUtils import AudioUtil
from soundClass import SoundClass

response = requests.get('https://firewave-models.s3.eu-central-1.amazonaws.com/model.pt')
open(Path.cwd() / 'model.pt', 'wb').write(response.content)
model = Model.load_from_file(Path.cwd() / 'model.pt')

summary(model, input_size=(2, 64, 126))

app = Flask(__name__)

@app.route('/')
def index():
    return 'It Works!!!'


@app.post('/upload')
def upload():
    file = request.files['file_data']
    tmp_file_name = f'data/{time()}_{random.randint(10000, 99999)}.wav'
    with wave.open(tmp_file_name, 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        waveform = file.stream.read()
        wavfile.writeframes(waveform)
    aud = AudioUtil.open(tmp_file_name)
    reaud = AudioUtil.resample(aud, 16000)
    rechan = AudioUtil.rechannel(reaud, 2)
    dur_aud = AudioUtil.pad_trunc(rechan, 4000)


    spectro_gram = AudioUtil.spectro_gram(dur_aud)
    inputs = spectro_gram[np.newaxis, ...]
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
    with torch.no_grad():
        outputs = model(inputs)
    print(f'Output: {outputs}')
    _, prediction = torch.max(outputs, 1)
    print(prediction.data[0])
    os.renames(tmp_file_name, f'data/archive/rec_{time()}__{random.randint(10000, 99999)}_{prediction.data[0]}.wav')
    return SoundClass(prediction.item()).name

# if __name__ == "__main__":
#     app.run()
