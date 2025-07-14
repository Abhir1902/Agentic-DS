from fastapi import FastAPI, BackgroundTasks, UploadFile, Form, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import subprocess
import threading
import os
import shutil
import zipfile
import time
import json

app = FastAPI()

PIPELINE_SCRIPT = 'agents/agents.py'
LOG_FILE = './logs/execution.log'
RESULT_NOTEBOOK = './solution/solution.ipynb'
MODEL_FILE = './model/best_model.pkl'
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'

pipeline_status = {'running': False, 'last_exit_code': None, 'last_message': ''}

# Run the pipeline in a background thread
def run_pipeline():
    pipeline_status['running'] = True
    pipeline_status['last_message'] = 'Pipeline started.'
    try:
        result = subprocess.run(['python3', PIPELINE_SCRIPT], capture_output=True, text=True)
        pipeline_status['last_exit_code'] = result.returncode
        pipeline_status['last_message'] = result.stdout + '\n' + result.stderr
    except Exception as e:
        pipeline_status['last_exit_code'] = -1
        pipeline_status['last_message'] = str(e)
    pipeline_status['running'] = False

@app.post('/run')
def run_agentic_pipeline(background_tasks: BackgroundTasks):
    if pipeline_status['running']:
        return JSONResponse({'status': 'already running'})
    background_tasks.add_task(run_pipeline)
    return {'status': 'started'}

@app.post('/kaggle_run')
def kaggle_run(background_tasks: BackgroundTasks, kaggle_link: str = Form(...), kaggle_username: str = Form(...), kaggle_key: str = Form(...)):
    if pipeline_status['running']:
        return JSONResponse({'status': 'already running'})
    # Write kaggle.json
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    kaggle_json = {
        "username": kaggle_username,
        "key": kaggle_key
    }
    with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
        json.dump(kaggle_json, f)
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    # Download competition data
    comp_name = kaggle_link.rstrip('/').split('/')[-1]
    try:
        subprocess.run(['kaggle', 'competitions', 'download', '-c', comp_name, '-p', './data'], check=True)
        # Unzip all files in ./data
        for file in os.listdir('./data'):
            if file.endswith('.zip'):
                with zipfile.ZipFile(f'./data/{file}', 'r') as zip_ref:
                    zip_ref.extractall('./data')
        # Move train/test files to correct directories if needed
        for file in os.listdir('./data'):
            if 'train' in file.lower() and file.endswith('.csv'):
                shutil.move(f'./data/{file}', f'{TRAIN_DIR}/{file}')
            elif 'test' in file.lower() and file.endswith('.csv'):
                shutil.move(f'./data/{file}', f'{TEST_DIR}/{file}')
    except Exception as e:
        return JSONResponse({'error': f'Kaggle download failed: {e}'}, status_code=500)
    background_tasks.add_task(run_pipeline)
    return {'status': 'started', 'competition': comp_name}

@app.get('/status')
def get_status():
    return pipeline_status

@app.get('/logs')
def get_logs():
    if os.path.exists(LOG_FILE):
        return FileResponse(LOG_FILE, media_type='text/plain')
    return JSONResponse({'error': 'Log file not found'}, status_code=404)

@app.get('/logs/stream')
def stream_logs():
    def log_streamer():
        last_size = 0
        while pipeline_status['running'] or os.path.exists(LOG_FILE):
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r') as f:
                    f.seek(last_size)
                    chunk = f.read()
                    if chunk:
                        yield chunk
                        last_size = f.tell()
            time.sleep(1)
            if not pipeline_status['running']:
                break
    return StreamingResponse(log_streamer(), media_type='text/plain')

@app.get('/notebook')
def get_notebook():
    if os.path.exists(RESULT_NOTEBOOK):
        return FileResponse(RESULT_NOTEBOOK, media_type='application/json')
    return JSONResponse({'error': 'Notebook not found'}, status_code=404)

@app.get('/model')
def get_model():
    if os.path.exists(MODEL_FILE):
        return FileResponse(MODEL_FILE, media_type='application/octet-stream')
    return JSONResponse({'error': 'Model file not found'}, status_code=404)

@app.get('/data/train')
def get_train_zip():
    zip_path = './data/train.zip'
    shutil.make_archive('./data/train', 'zip', TRAIN_DIR)
    return FileResponse(zip_path, media_type='application/zip', filename='train.zip')

@app.get('/data/test')
def get_test_zip():
    zip_path = './data/test.zip'
    shutil.make_archive('./data/test', 'zip', TEST_DIR)
    return FileResponse(zip_path, media_type='application/zip', filename='test.zip') 