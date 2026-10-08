# human_recognition

ROS 2 package for real-time face recognition using DeepFace.

## Features
- Detects and recognizes faces from a camera stream
- Matches against a local database of known faces (`database/<Name>/*.jpg`)
- Auto-registers unknown faces via terminal prompt (when `auto_register_unknown=true`)
- Publishes recognition results to `/face_recognition/result` (JSON)

## Nodes
| Node | Executable | Description |
|------|------------|-------------|
| `camera_publisher` | `camera_publisher` | Publishes camera images to `/camera/image_raw` |
| `face_recognizer` | `face_recognizer` | Runs face detection + recognition |
| `recognition_visualizer` | `recognition_visualizer` | Draws bounding boxes + names on image |

## Run

**All at once (recommended):**
```bash
ros2 launch human_recognition recognition.launch.py
```

**Separate terminals (for interactive unknown-face naming):**
```bash
# Terminal 1: Camera
ros2 run human_recognition camera_publisher

# Terminal 2: Recognition (prompts for name when unknown face detected)
ros2 run human_recognition face_recognizer

# Terminal 3 (optional): Visualization
ros2 run human_recognition recognition_visualizer
```

## Parameters (`face_recognizer`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera/image_raw` | Input image topic |
| `result_topic` | `/face_recognition/result` | Output topic |
| `db_path` | `<pkg>/database` | Path to face database |
| `model_name` | `VGG-Face` | Embedding model |
| `detector_backend` | `yunet` | Face detector |
| `recognition_hz` | `2.0` | Inference frequency |
| `auto_register_unknown` | `true` | Prompt to save unknown faces |

## Database Structure
```
database/
  PersonName/
    PersonName1.jpg
    PersonName2.jpg
    ...
```

Add folders with images to register new people.