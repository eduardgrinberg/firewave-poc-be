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
from settings import Settings

# response = requests.get('https://firewave-models.s3.eu-central-1.amazonaws.com/model.pt')
# open(Path.cwd() / 'model.pt', 'wb').write(response.content)
model = Model.load_from_file(Path.cwd() / 'model.pt')

summary(model, input_size=(1, 64, 126))

app = Flask(__name__)

lastStatus = False


@app.route('/')
def index():
    return 'It Works!!!'


def binary_prediction(output, threshold):
    return output > threshold


@app.post('/upload')
def upload():
    file = request.files['file_data']
    tmp_file_name = f'data/{time()}_{random.randint(10000, 99999)}.wav'
    with wave.open(tmp_file_name, 'wb') as wavfile:
        wavfile.setparams((1, Settings.bytes_per_sample, Settings.sample_rate, 0, 'NONE', 'NONE'))
        waveform = file.stream.read()
        wavfile.writeframes(waveform)
    aud = AudioUtil.open(tmp_file_name)
    dur_aud = AudioUtil.pad_trunc(aud, 4000)

    spectro_gram = AudioUtil.spectro_gram(dur_aud)
    inputs = spectro_gram[np.newaxis, ...]
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
    with torch.no_grad():
        outputs = model(inputs).squeeze(dim=0)
    print(f'Output: {outputs}')
    prediction = binary_prediction(outputs, Settings.threshold)
    print(prediction.data[0])
    file_name = f'{time()}__{random.randint(10000, 99999)}_{prediction.data[0]}.wav'
    os.renames(tmp_file_name, f'data/archive/{file_name}')

    lastStatus = prediction.item() == 1

    return jsonify({
        'fileName': file_name,
        'prediction': prediction.item() == 1,
        'score': outputs.item()
    })


@app.post('/feedback')
def feedback():
    request_data = request.get_json()
    file_name = request_data['fileName']
    correct_prediction = request_data['correctPrediction']
    class_prefix = 'fire' if correct_prediction else 'bg'
    os.renames(f'data/archive/{file_name}', f'data/feedback/{class_prefix}_{file_name}')
    return 'OK'

@app.get('/status')
def status():
    return str(lastStatus)
