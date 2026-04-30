from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse
import logging
import os
import sys
import signal
import threading
import base64
import torch
import pickle
import json
from io import BytesIO
import numpy as np
from typing import Optional

def setup_logging():
    """Setup logging configuration with LOG_DIR environment variable support"""
    # default log directory: atec/logs/
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.environ.get('LOG_DIR', os.path.join(project_dir, 'logs'))
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'user.log')
    
    # Create module-specific logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    if os.environ.get('LOG_TO_CONSOLE'):
        logger.addHandler(console_handler)
    
    return logger

# Setup logging
logger = setup_logging()


try:
    from solution import AlgSolution
    agent = AlgSolution()
except Exception as e:
    import traceback
    logger.error("Failed to initialize AlgSolution: %s", traceback.format_exc())
    exit(-1)

app = FastAPI()

logger.info("Server started")

@app.post('/step')
async def step(
    proprio: UploadFile = File(),
    extero: Optional[UploadFile] = File(None),
    head_rgb: Optional[UploadFile] = File(None),
    head_depth: Optional[UploadFile] = File(None),
    ee_rgb: UploadFile = File(),
    ee_depth: UploadFile = File(),
    video_rgb: Optional[UploadFile] = File(None),
    video_depth: Optional[UploadFile] = File(None),
    current_score: float= Form(),
):

    proprio = torch.tensor(np.frombuffer(await proprio.read(), dtype=np.float32).reshape(1, -1)).cuda()
    extero = torch.tensor(np.frombuffer(await extero.read(), dtype=np.float32).reshape(1, -1)).cuda() if extero is not None else None
    head_rgb = torch.tensor(np.frombuffer(await head_rgb.read(), dtype=np.uint8).reshape(1, 480, 640, 3)).cuda() if head_rgb is not None else None
    head_depth = torch.tensor(np.frombuffer(await head_depth.read(), dtype=np.float32).reshape(1, 480, 640, 1)).cuda() if head_depth is not None else None
    video_rgb = torch.tensor(np.frombuffer(await video_rgb.read(), dtype=np.uint8).reshape(1, 480, 640, 3)).cuda() if video_rgb is not None else None
    video_depth = torch.tensor(np.frombuffer(await video_depth.read(), dtype=np.float32).reshape(1, 480, 640, 1)).cuda() if video_depth is not None else None

    ee_rgb = torch.tensor(np.frombuffer(await ee_rgb.read(), dtype=np.uint8).reshape(1, 480, 640, 3)).cuda()
    ee_depth = torch.tensor(np.frombuffer(await ee_depth.read(), dtype=np.float32).reshape(1, 480, 640, 1)).cuda()

    if head_rgb is not None:
        obs = {
            'proprio': proprio,
            'extero': extero,
            'image': {
                'head_rgb': head_rgb,
                'head_depth': head_depth,
                'ee_rgb': ee_rgb,
                'ee_depth': ee_depth,
            }
        }
    else:
        obs = {
            'proprio': proprio,
            'extero': extero,
            'image': {
                'video_rgb': video_rgb,
                'video_depth': video_depth,
                'ee_rgb': ee_rgb,
                'ee_depth': ee_depth,
            }
        }
    action = agent.predicts(obs=obs, current_score=current_score)
    return action

@app.post('/reset')
async def reset(request: Request):
    form_data = await request.json()
    agent.reset(**form_data)
    return {"message": "success"}

@app.get('/synchronize')
async def synchronize():
    return {"message": "success"}

@app.get('/health')
async def health():
    return {"message": "success"}

@app.post('/stop')
async def stop(request: Request):
    body = await request.json()
    msg = body.get('msg')
    logger.info("Stop message received: %s", msg)
    return {"message": "success"}

@app.post('/quit')
async def quit(request: Request):
    """Gracefully shutdown the FastAPI application"""
    body = await request.json()
    msg = body.get('msg', 'quit')
    logger.info("Quit message received: %s", msg)

    # Use a timer to shutdown the server after sending response
    def shutdown_server():
        import uvicorn
        logger.info("Shutting down the server...")
        # This will send SIGTERM to the process
        os.kill(os.getpid(), signal.SIGTERM)

    # Start shutdown in a separate thread with a small delay to ensure response is sent
    shutdown_timer = threading.Timer(1.0, shutdown_server)
    shutdown_timer.start()

    return {"message": "Server is shutting down gracefully"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
