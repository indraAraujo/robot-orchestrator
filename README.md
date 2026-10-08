# Robô SAM

Sistema de controle embarcado do robô SAM do projeto de pesquisa realizado pela FEEC no laboratório [AdRoLab](https://almada.dca.fee.unicamp.br/adrolab/) desde 2026 orientado pelo Professor Doutor Eric Rohmer.

## Estrutura do Repositório

```
sam-robot/
├── src/
│   └── ros_ws/              # ROS 2 workspace
│       └── src/
│           └── human_recognition/   # Pacote de reconhecimento facial
├── README.md
└── ... (outros arquivos do projeto)
```

## Pacotes Atuais

### `human_recognition`
Reconhecimento facial em tempo real usando DeepFace.

**Funcionalidades:**
- Detecção e reconhecimento de rostos a partir de câmera
- Base de dados local de rostos conhecidos (`database/<Nome>/*.jpg`)
- Registro automático de rostos desconhecidos via prompt no terminal
- Publica resultados em `/face_recognition/result` (JSON)

**Nós:**
| Nó | Executável | Descrição |
|----|------------|-----------|
| `camera_publisher` | `camera_publisher` | Publica imagens da câmera em `/camera/image_raw` |
| `face_recognizer` | `face_recognizer` | Detecção + reconhecimento facial |
| `recognition_visualizer` | `recognition_visualizer` | Desenha caixas + nomes na imagem |

**Como executar:**
```bash
# Todos juntos (recomendado)
ros2 launch human_recognition recognition.launch.py

# Terminais separados (para nomear rostos desconhecidos interativamente)
# Terminal 1
ros2 run human_recognition camera_publisher
# Terminal 2
ros2 run human_recognition face_recognizer
# Terminal 3 (opcional)
ros2 run human_recognition recognition_visualizer
```

Veja mais detalhes em [`docs/human_recognition.md`](docs/human_recognition.md).
