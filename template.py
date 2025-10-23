import os
from src.Chatbot.logging import logger
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')



Project_name="Chatbot"

list_of_file=[
    '.github/workflow/.gitkeep',
    f"src/{Project_name}/__init__.py",
    f"src/{Project_name}/components/__init__.py",
    f"src/{Project_name}/utils/common.py"
    f"src/{Project_name}/research/research.ipynb"
    f"src/{Project_name}/utils/__init__.py",
    f"src/{Project_name}/logging/__init__.py",
    f"src/{Project_name}/config/__init__.py",
    f"src/{Project_name}/config/configration.py",
    f"src/{Project_name}/pipeline/__init__.py",
    f"src/{Project_name}/entity/__init__.py",
    f"src/{Project_name}/constants/__init__.py",
    "config/config.yaml",
    "param/params.yaml",
    "app.py",
    "main.py",
    ".gitignore",
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "setup.py"
]

for file_path in list_of_file:
    file_path=Path(file_path)
    filedir,filename=os.path.split(file_path)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
        logger.info(f"Creating Directory:{filedir} for the {filename}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path,'w') as f:
            pass
            logger.info(f"Creating empty file : {file_path}")
    
    else:
        pass
        logging.info(f"{filename} is Already exists")
    # add loog latter