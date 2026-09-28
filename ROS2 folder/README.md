Нода obstacle_detector принимает облако точек лидара, отдаёт каждый кадр алгоритму из папки core и публикует результат: есть ли препятствие на пути и на каком оно расстоянии. Всё работает в Docker (Ubuntu 22.04, ROS2 Humble). Нужны Docker, видеокарта NVIDIA и nvidia-container-toolkit. Записи лежат в папке data репозитория, внутри контейнера это /workspace/data.
### Запуск
Все команды выполняются из корня репозитория Obstacle-detection-system-with-a-3D-lidar.

Один раз собрать образ:
```bash
docker compose build
```
Дальше открываем четыре терминала.

Терминал 1, нода и RViz:
```bash
docker compose up demo
```
Терминал 2, полный результат по каждому кадру:
```bash
docker compose run --rm ros2 ros2 topic echo /obstacle/state obstacle_detector_msgs/msg/ObstacleState
```
Терминал 3, только сигнал тормозить или нет (true или false):
```bash
docker compose run --rm ros2 ros2 topic echo /obstacle/state obstacle_detector_msgs/msg/ObstacleState --field stop
```
Терминал 4, проигрывание записи с препятствием (запускать, когда RViz уже открылся):
```bash
docker compose run --rm ros2 ros2 bag play /workspace/data/scenarios/roundT_doubleT_approach_figure
```

Записи лежат в /workspace/data/for_hackathon/: doubleT_platform, roundT_doubleT, roundT_pressureGate_roundT, roundT_squareT_pressureGate_squareT, squareT_platform_squareT_switch.

Запись играется один раз, для повтора запустите команду ещё раз. Чтобы крутилась по кругу, допишите в конце --loop и на каждом новом круге нажимайте Reset в RViz внизу слева.

Если не хватает вычислительных мощностей, можно проигрывать запись медленнее, дописав --rate 0.5.

Терминалы 2, 3, 4 можно запускать и в уже работающем контейнере demo, без новых контейнеров: вместо docker compose run --rm ros2 пишите docker compose exec demo:
```bash
docker compose exec demo ros2 topic echo /obstacle/state obstacle_detector_msgs/msg/ObstacleState
docker compose exec demo ros2 topic echo /obstacle/state obstacle_detector_msgs/msg/ObstacleState --field stop
docker compose exec demo ros2 bag play /workspace/data/scenarios/roundT_doubleT_approach_figure
```

### Что смотреть
В /obstacle/state на каждый кадр приходит:
- header.stamp время кадра
- state 0 путь свободен, 1 не проверен, 2 препятствие
- stop нужно ли тормозить
- obstacle_found найдено ли препятствие
- nearest_distance_m расстояние до препятствия вдоль пути в метрах (nan, если его нет)
- checked_range_m на сколько метров вперёд путь проверен
- obstacles список препятствий: расстояние, центр, размер, число точек, уверенность

Чтобы смотреть одно поле, допишите к команде терминала 2 --field и имя поля, например --field nearest_distance_m.

В RViz видно облако лидара, синюю линию оси пути, зелёные границы проверенного участка и красный куб на месте препятствия. Если после смены записи разметка пропала, нажмите Reset внизу слева.
### Запись результатов
Пока играет запись, в отдельном терминале:
```bash
docker compose run --rm ros2 ros2 bag record -o /workspace/data/results/my_run /lidar_points /sensing/lidar/hesai128/pointcloud /obstacle/state /obstacle/markers /tf /tf_static
```

### Подписки и публикации
Нода слушает сразу два топика, /lidar_points и /sensing/lidar/hesai128/pointcloud (sensor_msgs/PointCloud2). Другой топик можно добавить в конфиг или указать при запуске. В облаке нужны поля x, y, z, intensity, желательно ещё ring и timestamp.

Публикует:
- /obstacle/state главный результат, тип obstacle_detector_msgs/ObstacleState. Одно сообщение на каждый кадр лидара, время в нём то же, что у кадра. Поля описаны выше в разделе Что смотреть.
- /obstacle/markers разметка для RViz (visualization_msgs/MarkerArray) в системе пути track: синяя линия оси пути, две зелёные линии границ проверенного участка и красный куб на каждое найденное препятствие. Каждый кадр старая разметка стирается и рисуется заново.
- /obstacle/cloud_preview копия облака лидара для RViz (sensor_msgs/PointCloud2) только с полями x, y, z, intensity и без пустых точек. Сколько точек в ней оставлять, задаёт preview_point_step.
- /tf положение системы пути track относительно лидара. Её каждый кадр находит алгоритм: ось X идёт вдоль пути вперёд, Z вверх. Публикуется, только когда путь найден.
- /tf_static положение лидара на поезде, base_link → hesai_lidar. Публикует launch-файл по аргументам lidar_x, lidar_y, lidar_z, поворот берётся из конфига алгоритма.

### TF
TF в ROS хранит, как системы координат расположены друг относительно друга. У нас их три, цепочкой:

base_link → hesai_lidar → track

- base_link поезд. От него считается всё остальное, в RViz это Fixed Frame.
- hesai_lidar лидар, в его координатах приходит облако. Где он стоит на поезде, задают lidar_x, lidar_y, lidar_z при запуске. Это положение не меняется, поэтому идёт в /tf_static.
- track путь впереди. Его каждый кадр заново находит алгоритм, поэтому он идёт в /tf. В этой системе рисуется разметка и считаются расстояния до препятствий.

Благодаря этой цепочке RViz сам ставит облако и разметку на свои места. Если стоит другой лидар, достаточно поменять lidar_frame и положение при запуске.
### Конфигурация
Все настройки в одном файле ROS2 folder/src/obstacle_detector/config/obstacle_detector.yaml. Сверху параметры ноды, под ключом core конфиг алгоритма. После правки перезапустите терминал 1.

Параметры ноды:
- pointcloud_topics список топиков, откуда брать облако, по умолчанию /lidar_points и /sensing/lidar/hesai128/pointcloud
- track_frame имя системы пути в TF и разметке, по умолчанию track
- lidar_period_s период кадров лидара в секундах, 0.1 для 10 Гц. По нему нода понимает, сколько кадров пропущено между двумя пришедшими, и сообщает это алгоритму.
- record_jump_s если между кадрами прошло больше этого времени (или время пошло назад), нода считает, что началась другая запись, и сбрасывает алгоритм. По умолчанию 1 с.
- required_range_m на сколько метров вперёд путь должен быть проверен, чтобы считаться свободным (state 0). Если проверено меньше, будет state 1. По умолчанию 0, то есть хватает любой проверенной дальности.
- preview_point_step сколько точек отдавать в RViz: 1 все, 2 каждую вторую и так далее. Если RViz тормозит, поставьте 2 или 4. На работу алгоритма не влияет.
- core настройки самого алгоритма, перенесены из его конфига без изменений.

Положение лидара на поезде задаётся при запуске аргументами lidar_x, lidar_y, lidar_z, по умолчанию 0, 0 и 1.081 м.
### Launch-файлы
Лежат в ROS2 folder/src/bringup/launch/:
- detector.launch.py нода и положение лидара на поезде
- rviz.launch.py RViz с конфигом bringup/config/lidar_monitor.rviz
- demo.launch.py оба сразу

Через Docker их запускают сервисы demo, detector и rviz:
```bash
docker compose up demo
docker compose up detector
docker compose up rviz
```
demo запускает по факту под собой и детектор и рвиз одновременно

Или вручную из консоли ROS (docker compose run --rm ros2):
```bash
ros2 launch bringup demo.launch.py
```
Аргументы пишутся через :=, например ros2 launch bringup demo.launch.py lidar_z:=1.5
- pointcloud_topic слушать только этот топик вместо списка из конфига, например pointcloud_topic:=/my_lidar/points
- lidar_frame frame_id облака, по умолчанию hesai_lidar
- lidar_x, lidar_y, lidar_z положение лидара на поезде в метрах
- bag путь к записи, чтобы она проигралась вместе с нодой
- detector_params свой конфиг вместо obstacle_detector.yaml
### Docker
Образ собирается из docker/Dockerfile, сервисы описаны в docker-compose.yml: demo нода и RViz, detector только нода, rviz только RViz, ros2 консоль. Папки ROS2 folder/src и data подключены в контейнер, поэтому правки кода и конфига работают без пересборки. Новые файлы и пакеты требуют docker compose build.

Перед запуском любого сервиса сам запускается служебный network-setup. Он поднимает на компьютере предел буфера сокета (net.core.rmem_max) до 64 МБ, иначе облака по 8 МБ частично теряются по пути к ноде. Общий буфер для остальных программ не меняется. Настройки DDS лежат в docker/cyclonedds.xml.

Окна (RViz) открываются через файл авторизации текущего сеанса, xhost не нужен. Если RViz всё же пишет, что не может подключиться к дисплею, выполните на компьютере xhost +local:docker.

ROS_DOMAIN_ID берётся из терминала, по умолчанию 0. Если на компьютере работает другой ROS и мешает, запускайте так: ROS_DOMAIN_ID=42 docker compose up demo, и в остальных терминалах с тем же номером.

### Запуск без docker compose
Если нужно именно docker build и docker run:
```bash
docker build -f docker/Dockerfile -t tunnel-lidar-ros2:humble .
sudo sysctl -w net.core.rmem_max=67108864
xhost +local:docker
docker run --rm -it --network host --ipc host --gpus all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/data:/workspace/data" tunnel-lidar-ros2:humble ros2 launch bringup demo.launch.py
```
sysctl здесь нужен руками, потому что без compose нет сервиса network-setup, и без него часть кадров теряется. Запись проигрывается из второго терминала:
```bash
docker run --rm -it --network host --ipc host -v "$(pwd)/data:/workspace/data" tunnel-lidar-ros2:humble ros2 bag play /workspace/data/scenarios/roundT_doubleT_approach_figure
```
