from roboflow import Roboflow
rf = Roboflow(api_key="7R6GP4tF3W16MZPfzxPX")
project = rf.workspace("eee-0ev7q").project("pcb-defect-0rfop")
version = project.version(1)
dataset = version.download("yolov11")
                