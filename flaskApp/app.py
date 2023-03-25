import os
import random
import wave
from pathlib import Path
from time import time

import numpy as np
import requests as requests
import torch
from flask import Flask, request, jsonify
from torchsummary import summary

from audioUtils import AudioUtil
from model import Model
from soundClass import SoundClass

response = requests.get('https://firewave-models.s3.eu-central-1.amazonaws.com/model.pt')
open(Path.cwd() / 'model.pt', 'wb').write(response.content)
model = Model.load_from_file(Path.cwd() / 'model.pt')

summary(model, input_size=(1, 64, 126))

app = Flask(__name__)


@app.route('/')
def index():
    return 'It Works!!!'


def binary_prediction(output):
    fire_predictions = output[:, 1]
    predictions = torch.zeros(fire_predictions.shape[0], dtype=torch.int)
    predictions[fire_predictions > 0] = 1

    return predictions


@app.post('/upload')
def upload():
    file = request.files['file_data']
    tmp_file_name = f'data/{time()}_{random.randint(10000, 99999)}.wav'
    with wave.open(tmp_file_name, 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        waveform = file.stream.read()
        wavfile.writeframes(waveform)
    aud = AudioUtil.open(tmp_file_name)
    dur_aud = AudioUtil.pad_trunc(aud, 4000)

    spectro_gram = AudioUtil.spectro_gram(dur_aud)
    inputs = spectro_gram[np.newaxis, ...]
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
    with torch.no_grad():
        outputs = model(inputs)
    print(f'Output: {outputs}')
    prediction = binary_prediction(outputs)
    print(prediction.data[0])
    file_name = f'{time()}__{random.randint(10000, 99999)}_{prediction.data[0]}.wav'
    os.renames(tmp_file_name, f'data/archive/{file_name}')

    return jsonify({
        'fileName': file_name,
        'prediction': prediction.item() == 1
    })


@app.post('/feedback')
def feedback():
    request_data = request.get_json()
    file_name = request_data['fileName']
    is_signal = request_data['isSignal']
    class_prefix = 'fire' if is_signal == 1 else 'bg'
    os.renames(f'data/archive/{file_name}', f'data/feedback/{class_prefix}_{file_name}')
    return 'OK'
