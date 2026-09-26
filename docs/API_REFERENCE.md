# API Reference: Ядро обнаружения препятствий

## Установка и окружение
- Python 3.10
- pip install -r requirements-core.txt (numpy, PyYAML, scipy)
- OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

## Быстрый старт
```
from core.config import Config
from core.pipeline import ObstacleDetector

detector = ObstacleDetector(Config.from_yaml("configs/default.yaml"))
result = detector.process(xyz, intensity, ring, t_rel, stamp_ns, frame_gap=1)
```

## Входные данные

| аргумент | пояснение | форма, тип |
|---|---|---|
| `xyz` | поля `x`, `y`, `z` | (N, 3), float32 |
| `intensity` | поле `intensity` | (N,), float32 |
| `ring` | поле `ring` | (N,), uint16 |
| `t_rel` | поле `timestamp` точек | (N,), float64 |
| `stamp_ns` | `header.stamp` в наносекундах | int |
| `frame_gap` | сколько кадров записи прошло с прошлого вызова | int |


## Выходные данные
| поле | смысл |
|---|---|
| obstacle_found | есть подтверждённое препятствие в габарите |
| nearest_distance_m | дистанция до ближайшего подтверждённого препятствия |
| detections | список обнаруженных препятствий |
| debug | диагностика |

## Detection
| поле | смысл |
| ---|---|
| distance_m | продольная координата x ближайшей точки кластера |
| centroid_xyz, bbox_min_xyz, bbox_max_xyz | центр и рамка кластера |
| n_points | точек в кластере |
| score | уверенность 0…1 |
| source | gauge |

## Что из debug нужно ноде
| ключ | смысл |
|---|---|
| checked | читать первым. False — кадр непригоден |
| frame_gap | сколько кадров прошло с прошлого проверенного |
| axis.state | measured / held / degraded / lost |
| axis.reason | почему ось не измерена |
| corridor.x_m | ближняя и дальняя границы проверенного участка |
| frame_transform | преобразование лидарных данных в путь |
| verdict | правило остановки при потере оси |

## Система координат
X — направлена вперед вдоль рельсов, Z — направлена вертикально вверх, а Y дополняет систему до правой, соответственно, направлена влево перпендикулярно оси X

## Пример без ROS
```
python examples/run_frames.py doubleT_platform --frames 5
python examples/run_frames.py doubleT_platform --frames 0 --jsonl out/example/doubleT_platform.jsonl
```

## Известные ограничения
- Ложные тревоги: 1.3/мин на пяти записях без препятствий
- Холодный старт: до 3.4с (0-34 кадра)
- Рабочая зона: 5-26м (медиана), до 51м с продлением
- Время кадра: 44-88мс (медиана), до 283мс (полный RANSAC)

## Проверка ноды
Ядро детерминировано. Одна и та же запись через ноду должна дать ровно те же числа. <br>
Позитив: roundT_doubleT_approach_figure, 252 кадра, ожидается 174 кадра с obstacle_found <br>
Негатив: doubleT_platform, 345 кадров, ожидается 4 кадра с obstacle_found (24, 27, 28, 29)

