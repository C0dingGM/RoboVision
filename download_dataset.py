from roboflow import Roboflow
import os

api_key = os.getenv('ROBOFLOW_API_KEY', 'nNqhATNXCFnQLeggRpIc')
rf = Roboflow(api_key=api_key)

project = rf.workspace("wangs-workspace-ahzkf").project("chess-full-yc6fm")
version = project.version(1)
dataset = version.download("yolov11")

print(f"\nDataset downloaded to: {dataset.location}")
