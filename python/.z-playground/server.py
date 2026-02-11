from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/calculate', methods=['POST'])
def calculate():
    time = float(request.json['time'])
    length = 1  # Pendulum length
    g = 9.81  # Gravity
    theta = np.pi / 4 * np.cos(np.sqrt(g / length) * time)
    
    x = length * np.sin(theta)
    y = -length * np.cos(theta)
    return jsonify({'x': x, 'y': y})

if __name__ == '__main__':
    app.run()