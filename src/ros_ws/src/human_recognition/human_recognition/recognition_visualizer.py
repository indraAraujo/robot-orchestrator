import json
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

COLOR_KNOWN = (0, 220, 0)
COLOR_UNKNOWN = (0, 165, 255)
COLOR_IDLE = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class RecognitionVisualizerNode(Node):
    def __init__(self):
        super().__init__('recognition_visualizer_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('result_topic', '/face_recognition/result')
        self.declare_parameter('output_topic', '/face_recognition/image')
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'SAM - reconhecimento')
        self.declare_parameter('max_fps', 30.0)
        self.declare_parameter('box_thickness', 2)
        self.declare_parameter('label_scale', 0.6)

        image_topic = self.get_parameter('image_topic').value
        result_topic = self.get_parameter('result_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = self.get_parameter('window_name').value
        self.max_fps = max(float(self.get_parameter('max_fps').value), 1.0)
        self.thickness = int(self.get_parameter('box_thickness').value)
        self.label_scale = float(self.get_parameter('label_scale').value)

        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, output_topic, 10)

        self._lock = threading.Lock()
        self._result = None
        self._result_time = 0.0
        self._frames = 0
        self._fps = 0.0
        self._fps_window_start = time.time()
        self._fps_window_frames = 0
        self._stop = threading.Event()
        self._window_frame = None
        self._window_frame_ready = threading.Event()

        self.create_subscription(String, result_topic, self.result_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        self.get_logger().info(
            f'Visualizador iniciado: {image_topic} + {result_topic} -> {output_topic}'
            + (f' (janela "{self.window_name}")' if self.show_window else '')
        )

        if self.show_window:
            threading.Thread(target=self._window_loop, daemon=True).start()

    def result_callback(self, msg: String) -> None:
        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Resultado ignorado (JSON invalido): {exc}')
            return
        with self._lock:
            self._result = result
            self._result_time = time.time()

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(
                f'Nao foi possivel converter a imagem: {exc}', throttle_duration_sec=5.0)
            return

        with self._lock:
            result = self._result
            age = time.time() - self._result_time

        canvas = self._annotate(frame, result, age)

        out = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id or 'camera_link'
        self.publisher_.publish(out)

        if self.show_window:
            with self._lock:
                self._window_frame = canvas
            self._window_frame_ready.set()

    def _annotate(self, frame: np.ndarray, result, age: float) -> np.ndarray:
        self._frames += 1
        now = time.time()
        self._fps_window_frames += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps = self._fps_window_frames / elapsed
            self._fps_window_frames = 0
            self._fps_window_start = now

        height, width = frame.shape[:2]
        scale = max(width / 640.0, 0.5)

        fresh = result is not None and age <= 2.0

        if fresh and result.get('detected') and result.get('box'):
            identity = result.get('identity', 'Unknown')
            known = identity != 'Unknown'
            color = COLOR_KNOWN if known else COLOR_UNKNOWN

            x, y, w, h = result['box']
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = min(int(x) + int(w), width)
            y2 = min(int(y) + int(h), height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            confidence = result.get('confidence')
            label = identity
            if confidence is not None and confidence == confidence:
                label += f'  {confidence:.1f}%'
            self._draw_label(frame, label, (x1, y1), color, scale)
        else:
            color = COLOR_IDLE
            text = 'procurando rostos...' if fresh else 'sem resultado'
            self._draw_banner(frame, text, color, scale)

        self._draw_banner(frame, f'{self._fps:4.1f} FPS', COLOR_IDLE, scale, top=False)
        return frame

    def _draw_banner(self, frame, text, color, scale, top=True) -> None:
        height = frame.shape[0]
        y = 22 if top else height - 12
        cv2.putText(frame, text, (8, y), FONT, 0.5 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (8, y), FONT, 0.5 * scale, color, 1, cv2.LINE_AA)

    def _draw_label(self, frame, text, origin, color, scale) -> None:
        font_scale = 0.55 * scale
        thickness = max(int(round(self.thickness)), 1)
        (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
        x, y = origin
        ty = max(y - 8, th + 8)
        cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 10, ty + baseline), color, -1)
        cv2.putText(frame, text, (x + 5, ty), FONT, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)

    def _window_loop(self) -> None:
        try:
            while rclpy.ok() and not self._stop.is_set():
                if not self._window_frame_ready.wait(timeout=0.5):
                    continue
                self._window_frame_ready.clear()
                with self._lock:
                    frame = self._window_frame
                if frame is None:
                    continue
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    self.get_logger().info('Fechando a janela (q)')
                    self._stop.set()
                    break
        except Exception as exc:
            self.get_logger().warn(f'Janela local indisponivel: {exc}')
        finally:
            cv2.destroyAllWindows()

    def destroy_node(self):
        self._stop.set()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecognitionVisualizerNode()
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
