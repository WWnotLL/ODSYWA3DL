## Рекомендации
Рекомендуется использовать машину/контейнер с Ubuntu 22.04 и установленным ROS2 Humble

## Как разархивировать
Используя команду:
```bash
tar --zstd -xvf for_hackathon.zst
```

## Как проиграть бэги
Используя команду (на примере бэга "doubleT_platform") из той же директории, где лежит директория "for_hackathon":
```bash
ros2 bag play for_hackathon/doubleT_platform
```