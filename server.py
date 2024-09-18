from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run_pyqt', methods=['GET'])
def run_pyqt():
    try:
        subprocess.Popen(['python', '3.py'])  # Run your PyQt5 application
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
