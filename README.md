# Human Pose (VS Code)

Uses OpenCV DNN with the included OpenPose COCO Caffe model:
- `pose_deploy_linevec.prototxt`
- `pose_iter_440000.caffemodel`

## Setup

```powershell
cd "c:\Users\ayush\Documents\human pos"
python -m pip install -r requirements.txt
```

## Run (webcam)

```powershell
python .\app.py --source webcam --flip
```

If the webcam fails to open, try another camera index:

```powershell
python .\app.py --source webcam --flip --cam 1
```

If it still fails, try scanning more camera indices:

```powershell
python .\app.py --source webcam --flip --max-cams 20
```

## Run (video file)

```powershell
python .\app.py --source ".\my_video.mp4"
```

## Controls

- Press `q` or `Esc` to quit.

## What it shows

- Keypoints + skeleton lines
- Labels: `Standing` / `Sitting`, `Left/Right/Both Hands Raised`, `Not Looking`
- Movement: `Move: <pixels>` + direction

