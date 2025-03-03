# Crowd_detection

Проект по детекции людей в толпе с использованием модели семейства YOLO.

## Описание проекта

Проект предназначен для обнаружения людей в толпе с использованием моделей YOLO. Модель была дообучена на собственном датасете, размеченном в CVAT. В проекте также используется алгоритм SORT для отслеживания объектов, что позволяет улучшить точность и стабильность детекции. Результаты обучения и оценки моделей представлены в файле `crowd_detection.ipynb`.

## Установка и запуск

Для запуска программы необходимо выполнить следующие шаги:

0. Клонировать репозиторий
    ```bash
    git clone https://github.com/daniluck505/Crowd_detection.git\
    cd Crowd_detection
    ```
1. Установите необходимые библиотеки из файла `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
2. Настройте конфигурационный файл `config.yml` в соответствии с вашими требованиями.
3. Запустите скрипт `crowd_detection.py`:
    ```bash
    python crowd_detection.py
    ```

## Примеры работы
<video controls autoplay loop src="https://github.com/daniluck505/Crowd_detection/blob/master/videos/out/final.mp4" width="100%"></video>


