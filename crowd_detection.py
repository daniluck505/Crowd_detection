import torch
import numpy as np
import cv2
from ultralytics import YOLO
import time
import os
import random
import yaml
from sort import Sort


class VideoBuilder:
    """
    Класс для построения и обработки видео с использованием различных трансформаций.
    Атрибуты:
        transforms (list): Список функций-трансформаций, применяемых к каждому кадру.
        fps_counter (FPS_Counter): Счетчик FPS для отображения на кадре.
        sort_tracker (Sort): Трекер для отслеживания объектов с использованием алгоритма SORT.
    """
    def __init__(self):
        self.transforms = []
        self.fps_counter = None
        self.sort_tracker = None
    
    def resize(self, scale=0.5):
        """
        Добавляет трансформацию изменения размера кадра.
        Аргументы:
            scale (float): Масштаб изменения размера кадра. По умолчанию 0.5.
        Возвращает:
            self: Возвращает экземпляр класса для цепочки вызовов.
        """
        self.transforms.append(lambda frame: cv2.resize(frame, (-1, -1), fx=scale, fy=scale))
        return self

    def fps(self, calc_time_perion_N_frames=10):
        """
        Добавляет трансформацию для отображения FPS на кадре.
        Аргументы:
            calc_time_perion_N_frames (int): Количество кадров для расчета FPS. По умолчанию 10.
        Возвращает:
            self: Возвращает экземпляр класса для цепочки вызовов.
        """
        if not self.fps_counter:
            self.fps_counter = FPS_Counter(calc_time_perion_N_frames)
        
        def display_fps(frame):
            fps_real = self.fps_counter.calc_FPS()
            text = f"FPS: {fps_real:.1f}"

            fontFace = 1
            fontScale = 1.3
            thickness = 1
            
            (label_width, label_height), _ = cv2.getTextSize(
                text,
                fontFace=fontFace,
                fontScale=fontScale,
                thickness=thickness,
            )
            frame = cv2.rectangle(frame, (0, 0), (10 + label_width, 15 + label_height), (0, 0, 0), -1)
            frame = cv2.putText(
                img=frame,
                text=text,
                org=(5, 20),
                fontFace=fontFace,
                fontScale=fontScale,
                thickness=thickness,
                color=(255, 255, 255),
            )
            return frame
        
        self.transforms.append(display_fps)
        return self

    def detection(self, model, imgsz=640, conf=0.7, iou=0.7, show_text=True, only_person=True):
        """
        Добавляет трансформацию для обнаружения объектов на кадре.
        Аргументы:
            model: Модель для обнаружения объектов.
            imgsz (int): Размер изображения для модели. По умолчанию 640.
            conf (float): Порог уверенности для обнаружения. По умолчанию 0.7.
            iou (float): Порог IoU для NMS. По умолчанию 0.7.
            show_text (bool): Флаг для отображения текста на кадре. По умолчанию True.
            only_person (bool): Флаг для обнаружения только людей. По умолчанию True.
        Возвращает:
            self: Возвращает экземпляр класса для цепочки вызовов.
        """
        self.transforms.append(lambda frame: self.__visualize_detection(
                                            frame,
                                            model,
                                            imgsz,
                                            conf,
                                            iou,
                                            show_text=show_text,
                                            only_person=only_person
                                        ))
        return self

    def __visualize_detection(self, img, model, imgsz, conf, iou, show_text, only_person):
        """
        Визуализирует обнаруженные объекты на кадре.
        Аргументы:
            img: Входной кадр.
            model: Модель для обнаружения объектов.
            imgsz (int): Размер изображения для модели.
            conf (float): Порог уверенности для обнаружения.
            iou (float): Порог IoU для NMS.
            show_text (bool): Флаг для отображения текста на кадре.
            only_person (bool): Флаг для обнаружения только людей.
        Возвращает:
            img: Кадр с визуализированными объектами.
        """
        predictions = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

        labeled_image = img.copy()

        if self.sort_tracker:
            return self.__plot_bboxes_with_sort(labeled_image, predictions, show_text, only_person)
        else:
            class_names = model.names
            return self.__plot_bboxes_with_class(labeled_image, predictions, class_names, show_text, only_person)
    
    def __plot_bboxes_with_class(self, frame, predictions, class_names, show_class, only_person):
        """
        Рисует bounding box'ы с классами на кадре.
        Аргументы:
            frame: Входной кадр.
            predictions: Предсказания модели.
            class_names: Список имен классов.
            show_class (bool): Флаг для отображения класса на bounding box'е.
            only_person (bool): Флаг для отображения только людей.
        Возвращает:
            frame: Кадр с bounding box'ами.
        """
        for pred in predictions:
            boxes = pred.boxes.xyxy.cpu().int().tolist()
            classes = pred.boxes.cls.cpu().int().tolist()
            confidences = pred.boxes.conf.cpu().numpy()

            for i, (box, class_index) in enumerate(zip(boxes, classes)):
                if only_person and int(class_index) != 0:
                    continue
                class_name = class_names[int(class_index)]
                random.seed(int(classes[i])+10)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                x_min, y_min, x_max, y_max = box

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 4)

                if show_class:
                    label = f'{class_name} {confidences[i]:.2f}'
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x_min, y_min - text_height - 10), (x_min + text_width, y_min), color, -1)
                    cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def __plot_bboxes_with_sort(self, frame, predictions, show_ids, only_person):
        """
        Рисует bounding box'ы с использованием трекера SORT.
        Аргументы:
            frame: Входной кадр.
            predictions: Предсказания модели.
            show_ids (bool): Флаг для отображения ID объектов.
            only_person (bool): Флаг для отображения только людей.
        Возвращает:
            frame: Кадр с bounding box'ами и ID объектов.
        """
        detections_list = []
        for pred in predictions:
            boxes = pred.boxes.xyxy.cpu().int().tolist()
            classes = pred.boxes.cls.cpu().int().tolist()
            confidences = pred.boxes.conf.cpu().numpy()

            for i, (box, class_index) in enumerate(zip(boxes, classes)):
                if only_person and int(class_index) != 0:
                    continue
                x_min, y_min, x_max, y_max = box
                detections_list.append([x_min, y_min, x_max, y_max, confidences[i]])

        if len(detections_list) == 0:
                detections_list = np.empty((0, 5))

        res = self.sort_tracker.update(np.array(detections_list))
        for track in res:
            x_min, y_min, x_max, y_max, track_id = track
            random.seed(15)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 4)
            if show_ids:
                cv2.putText(frame, f"ID: {int(track_id)}", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    def sort(self, max_age=100, min_hits=8, iou_threshold=0.50, show_ids=True):
        """
        Инициализирует трекер SORT.
        Аргументы:
            max_age (int): Максимальное количество кадров для сохранения трека без обновления. По умолчанию 100.
            min_hits (int): Минимальное количество попаданий для инициализации трека. По умолчанию 8.
            iou_threshold (float): Порог IoU для сопоставления треков. По умолчанию 0.50.
            show_ids (bool): Флаг для отображения ID объектов. По умолчанию True.
        Возвращает:
            self: Возвращает экземпляр класса для цепочки вызовов.
        """
        self.sort_tracker = Sort(max_age=max_age, 
                                 min_hits=min_hits, 
                                 iou_threshold=iou_threshold) 
        self.show_ids = show_ids
        return self

    def stream_camera(self, camera=0, size=(720, 480)):
        """
        Запускает потоковое видео с веб-камеры с применением всех трансформаций.

        Аргументы:
            camera (int): Индекс камеры. По умолчанию 0.
            size (tuple): Размер кадра (ширина, высота). По умолчанию (720, 480).
        """
        cap = cv2.VideoCapture(camera)
        assert cap.isOpened(), 'Не удалось открыть камеру'
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

        while True:
            ret, frame = cap.read()
            assert ret, 'Не удалось получить кадр с веб-камеры'
            
            transformed_frame = frame.copy()
            for transform in self.transforms:
                transformed_frame = transform(transformed_frame)            
            
            cv2.imshow('Webcam', transformed_frame)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def stream_video(self, video_path, video_out_path=None, out_name='out.mp4'):
        """
        Запускает потоковое видео из файла с применением всех трансформаций.

        Аргументы:
            video_path (str): Путь к видеофайлу.
            video_out_path (str): Путь для сохранения обработанного видео. По умолчанию None.
            out_name (str): Имя выходного файла. По умолчанию 'out.mp4'.
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if video_out_path:
            for transform in self.transforms:
                frame = transform(frame)
            video_out_path = os.path.join(video_out_path, out_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            frame_size = (int(frame_width), int(frame_height))

            cap_out = cv2.VideoWriter(video_out_path, 
                                    fourcc, 
                                    fps,
                                    frame_size)
        
        while ret:
            transformed_frame = frame.copy()
            for transform in self.transforms:
                transformed_frame = transform(transformed_frame)

            cv2.imshow(video_path, transformed_frame)
            if video_out_path:
                cap_out.write(transformed_frame)
            ret, frame = cap.read()

            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        
        
        cap.release()
        if video_out_path:
            cap_out.release()
        cv2.destroyAllWindows()


class FPS_Counter:
    """
    Класс для подсчета FPS (количество кадров в секунду).

    Атрибуты:
        time_buffer (list): Буфер для хранения временных меток.
        calc_time_perion_N_frames (int): Количество кадров для расчета FPS.
    """
    def __init__(self, calc_time_perion_N_frames):
        self.time_buffer = []
        self.calc_time_perion_N_frames = calc_time_perion_N_frames

    def calc_FPS(self):
        time_buffer_is_full = len(self.time_buffer) == self.calc_time_perion_N_frames
        t = time.time()
        self.time_buffer.append(t)

        if time_buffer_is_full:
            self.time_buffer.pop(0)
            fps = len(self.time_buffer) / (self.time_buffer[-1] - self.time_buffer[0])
            return np.round(fps, 2)
        else:
            return 0.0
        

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


with open('config.yml', 'r') as f:
    options = yaml.safe_load(f)

video_path = options['video_path']
save_path = options['save_path']
save_name = options['save_name']
model_weights = options['model_weights']
imgsz = options['imgsz']
conf = options['conf']
iou = options['iou'] 
max_age = options['max_age']
min_hits = options['min_hits']
iou_threshold = options['iou_threshold']
show_ids = options['show_ids']
sort = options['sort']


stream = VideoBuilder()
model = YOLO(model_weights)

if sort:
    stream.detection(model, imgsz, conf, iou, True).sort(max_age, min_hits, iou_threshold, show_ids).stream_video(video_path, save_path, save_name)
else:
    stream.detection(model, imgsz, conf, iou, True).stream_video(video_path, save_path, save_name)


