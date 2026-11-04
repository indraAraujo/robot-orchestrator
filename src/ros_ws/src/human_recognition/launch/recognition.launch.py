from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    show_window = LaunchConfiguration('show_window')

    return LaunchDescription([
        DeclareLaunchArgument(
            'show_window',
            default_value='true',
            description='Abre a janela OpenCV local alem de publicar o topico'),
        DeclareLaunchArgument(
            'recognition_hz',
            default_value='2.0',
            description='Frequencia de inferencia do deepface'),
        Node(
            package='human_recognition',
            executable='camera_publisher',
            name='camera_publisher_node',
            output='screen',
        ),
        Node(
            package='human_recognition',
            executable='face_recognizer',
            name='face_recognizer_node',
            output='screen',
            parameters=[{'recognition_hz': LaunchConfiguration('recognition_hz')}],
        ),
        Node(
            package='human_recognition',
            executable='recognition_visualizer',
            name='recognition_visualizer_node',
            output='screen',
            parameters=[{'show_window': show_window}],
        ),
    ])
