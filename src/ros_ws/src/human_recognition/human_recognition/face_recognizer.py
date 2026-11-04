import json
import logging
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String

os.environ.setdefault('DEEPFACE_LOG_LEVEL', str(logging.WARNING))

from deepface import DeepFace  
from deepface.modules.exceptions import FaceNotDetected  
from deepface.modules import detection  

DEFAULT_MODEL = 'VGG-Face'
DEFAULT_DETECTOR = 'yunet'
RESULT_UNKNOWN = 'Unknown'


def default_db_path() -> str:
    installed = os.path.join(get_package_share_directory('human_recognition'), 'database')
    if os.path.isdir(installed):
        return installed
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')


def _identity_name(identity: str) -> str:
    folder = os.path.basename(os.path.dirname(str(identity)))
    return folder or str(identity)


class FaceRecognizerNode(Node):

    def __init__(self):
        super().__init__('face_recognizer_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('result_topic', '/face_recognition/result')
        self.declare_parameter('db_path', default_db_path())
        self.declare_parameter('model_name', DEFAULT_MODEL)
        self.declare_parameter('detector_backend', DEFAULT_DETECTOR)
        self.declare_parameter('recognition_hz', 2.0)
        self.declare_parameter('threshold', 0.0)
        self.declare_parameter('enforce_detection', True)
        self.declare_parameter('warmup_frames', 10)
        self.declare_parameter('last_identity', RESULT_UNKNOWN)
        self.declare_parameter('auto_register_unknown', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.db_path = self.get_parameter('db_path').value
        self.model_name = self.get_parameter('model_name').value
        self.detector_backend = self.get_parameter('detector_backend').value
        self.enforce_detection = self.get_parameter('enforce_detection').value
        self.warmup_frames = int(self.get_parameter('warmup_frames').value)
        self.auto_register_unknown = self.get_parameter('auto_register_unknown').value

        threshold = float(self.get_parameter('threshold').value)
        self.threshold = None if threshold <= 0.0 else threshold

        if not os.path.isdir(self.db_path):
            self.get_logger().error(
                f"Base de rostos nao encontrada em '{self.db_path}'. "
                'Esperado: <db_path>/<Nome>/<imagens>.'
            )

        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(String, self.result_topic, 10)

        self._lock = threading.Lock()
        self._latest_frame = None
        self._frames_seen = 0
        self._refresh_database = True
        self._first_inference_done = False
        self._pending_unknown = None
        self._unknown_lock = threading.Lock()

        self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        rate = max(float(self.get_parameter('recognition_hz').value), 0.1)
        self.create_timer(1.0 / rate, self.timer_callback)

        self.get_logger().info(
            f'Recognizer iniciado: modelo={self.model_name} detector={self.detector_backend} '
            f'db={self.db_path} topic={self.image_topic} -> {self.result_topic} ({rate} Hz)'
        )
        self._build_model()

    def _build_model(self) -> None:
        started = time.time()
        self.get_logger().info('Carregando pesos do modelo (a primeira vez pode demorar)')
        try:
            DeepFace.build_model(model_name=self.model_name)
            DeepFace.build_model(task='face_detector', model_name=self.detector_backend)
        except Exception as exc:
            self.get_logger().error(f'Falha ao carregar o modelo: {exc}')
            raise
        self.get_logger().info(f'Modelo pronto em {time.time() - started:.1f}s')

    def image_callback(self, msg: Image) -> None:
        self._frames_seen += 1
        if self._frames_seen <= self.warmup_frames:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Nao foi possivel converter a imagem: {exc}', throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest_frame = frame

    def timer_callback(self) -> None:
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        if frame is None:
            return

        match = self._recognize(frame)
        if match is None:
            return

        self._publish(match)

        if match.get('face_found', False):
            if match['identity'] != RESULT_UNKNOWN:
                self.get_logger().info(
                    f"Rosto reconhecido: {match['identity']} "
                    f"(distancia {match['distance']:.4f}, confianca {match['confidence']:.1f}%)",
                    throttle_duration_sec=2.0)
            else:
                if self.auto_register_unknown:
                    self._handle_unknown_face(frame, match)

    def _publish(self, match: dict) -> None:
        msg = String()
        msg.data = json.dumps(match, separators=(',', ':'))
        self.publisher_.publish(msg)
        self.set_parameters([Parameter('last_identity', value=match['identity'])])

    def _log_first_inference(self, started: float) -> None:
        if self._first_inference_done:
            return
        self._first_inference_done = True
        self.get_logger().info(
            f'Primeira inferencia concluida em {time.time() - started:.1f}s '
            f'(inclui codificar a base de rostos)')

    def _detect_face_box(self, frame: np.ndarray):
        try:
            face_objs = detection.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False,
            )
            if face_objs:
                facial_area = face_objs[0]['facial_area']
                return [facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']]
        except Exception as exc:
            self.get_logger().warn(f'Falha ao detectar rosto para bbox: {exc}')
        return None

    def _recognize(self, frame: np.ndarray):
        started = time.time()
        try:
            results = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                threshold=self.threshold,
                silent=True,
                refresh_database=self._refresh_database,
            )
        except FaceNotDetected:
            self._log_first_inference(started)
            self.get_logger().info(
                'Nenhum rosto detectado no frame', throttle_duration_sec=2.0)
            return {'detected': False, 'identity': RESULT_UNKNOWN, 'box': None, 'face_found': False}
        except ValueError as exc:
            self.get_logger().error(f'Erro do deepface: {exc}', throttle_duration_sec=5.0)
            return None
        finally:
            self._refresh_database = False

        self._log_first_inference(started)

        if not results:
            self.get_logger().info(
                'Rosto detectado mas nao identificado', throttle_duration_sec=2.0)
            box = self._detect_face_box(frame)
            return {'detected': True, 'identity': RESULT_UNKNOWN, 'box': box, 'face_found': True, 'distance': float('inf')}

        for df in results:
            if df is None or df.empty:
                continue
            row = df.iloc[0]
            distance = row.get('distance', row.get('face_dist'))
            if distance is None:
                continue
            return {
                'detected': True,
                'identity': _identity_name(row['identity']),
                'distance': float(distance),
                'confidence': float(row.get('confidence', float('nan'))),
                'box': [int(row['source_x']), int(row['source_y']),
                        int(row['source_w']), int(row['source_h'])],
                'face_found': True,
            }

        self.get_logger().info(
            'Rosto detectado mas nao identificado', throttle_duration_sec=2.0)
        box = self._detect_face_box(frame)
        return {'detected': True, 'identity': RESULT_UNKNOWN, 'box': box, 'face_found': True, 'distance': float('inf')}

    def _get_next_image_number(self, person_folder: str) -> int:
        if not os.path.isdir(person_folder):
            return 1
        existing = []
        for fname in os.listdir(person_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    num = int(''.join(filter(str.isdigit, fname)))
                    existing.append(num)
                except ValueError:
                    pass
        return max(existing) + 1 if existing else 1

    def _save_unknown_face(self, frame: np.ndarray, box: list, name: str) -> str:
        person_folder = os.path.join(self.db_path, name)
        os.makedirs(person_folder, exist_ok=True)

        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        face_crop = frame[y:y+h, x:x+w]

        next_num = self._get_next_image_number(person_folder)
        filename = f"{name}{next_num}.jpg"
        filepath = os.path.join(person_folder, filename)

        cv2.imwrite(filepath, face_crop)
        self.get_logger().info(f'Rosto salvo como {filepath}')
        return filepath

    def _prompt_for_name(self, frame: np.ndarray, match: dict) -> None:
        def ask_name():
            try:
                name = input('\n[Registro] Digite o nome da pessoa nao reconhecida: ').strip()
                if not name:
                    self.get_logger().info('Nome vazio, ignorando registro.')
                    return
                name = ''.join(word.capitalize() for word in name.split())
                with self._unknown_lock:
                    self._pending_unknown = None
                self._save_unknown_face(frame, match['box'], name)
                self._refresh_database = True
                self.get_logger().info(f'Base de dados atualizada com nova pessoa: {name}')
            except (EOFError, KeyboardInterrupt):
                self.get_logger().info('Registro cancelado.')
                with self._unknown_lock:
                    self._pending_unknown = None

        thread = threading.Thread(target=ask_name, daemon=True)
        thread.start()

    def _handle_unknown_face(self, frame: np.ndarray, match: dict) -> None:
        with self._unknown_lock:
            if self._pending_unknown:
                return
            self._pending_unknown = (frame.copy(), match)

        self.get_logger().info('Rosto desconhecido detectado. Aguardando nome para registro...')
        self._prompt_for_name(frame, match)


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
