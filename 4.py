from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-script', methods=['POST'])
def run_script():
    # This will run the Python script
    subprocess.Popen(['python', '3.py'], shell=True)
    return "Script is running!"

if __name__ == '__main__':
    app.run(debug=True)
