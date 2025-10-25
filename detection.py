### Импорт необходимых библиотек 
import cv2  
import numpy as np  
from PIL import Image, ImageDraw, ImageFont  
import io  
from typing import List, Tuple, Dict  
from ultralytics import YOLO 
import logging  

### Создание логгера для отслеживания работы модуля
logger = logging.getLogger(__name__)

### Детекция и сегментация объектов на изображении 
class YOLOObjectDetector:
    """
    Класс для детекции и сегментации объектов на изображениях с использованием YOLO
    
    Основные функции:
    - Детекция объектов с отрисовкой ограничивающих рамок
    - Сегментация объектов с созданием цветных масок
    - Предобработка изображений для оптимальной работы модели
    - Перевод названий классов на русский язык
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', seg_model_name: str = 'yolov8n-seg.pt', confidence_threshold: float = 0.5):
        """
        Инициализация детектора объектов
        
        Args:
            model_name: Название модели YOLO для детекции (yolov8n.pt)
            seg_model_name: Название модели YOLO для сегментации (yolov8n-seg.pt)
            confidence_threshold: Порог уверенности для детекции (0.0-1.0, по умолчанию 0.5)
        
        Используем nano-модели для обеих задач, как самые быстрые
        """
        ## Настройки модели и обработки изображений
        self.confidence_threshold = confidence_threshold  # Минимальная уверенность для принятия детекции
        self.max_image_size = 1280  # Максимальный размер изображения для ускорения обработки
        self.min_object_area = 400  # Минимальная площадь объекта в пикселях для фильтрации мелких объектов
        
        ## Загрузка предобученных моделей YOLO
        try:
            # Загружаем модель для детекции объектов (определение класса и bboxes)
            logger.info(f"Загружаю модель YOLO для детекции: {model_name}")
            self.model = YOLO(model_name)
            logger.info("Модель YOLO для детекции успешно загружена")
            
            # Загружаем модель для сегментации объектов 
            logger.info(f"Загружаю модель YOLO для сегментации: {seg_model_name}")
            self.seg_model = YOLO(seg_model_name)
            logger.info("Модель YOLO для сегментации успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей YOLO: {e}")
            raise
        
        ## Словарь для перевода английских названий основных классов COCO на русский язык (на датасете COCO обучаются YOLO модели)
        self.class_translations = {
            'person': 'человек',
            'bicycle': 'велосипед', 
            'car': 'автомобиль',
            'motorcycle': 'мотоцикл',
            'airplane': 'самолет',
            'bus': 'автобус',
            'train': 'поезд',
            'truck': 'грузовик',
            'boat': 'лодка',
            'traffic light': 'светофор',
            'fire hydrant': 'пожарный гидрант',
            'stop sign': 'знак стоп',
            'parking meter': 'паркомат',
            'bench': 'скамейка',
            'bird': 'птица',
            'cat': 'кот',
            'dog': 'собака',
            'horse': 'лошадь',
            'sheep': 'овца',
            'cow': 'корова',
            'elephant': 'слон',
            'bear': 'медведь',
            'zebra': 'зебра',
            'giraffe': 'жираф',
            'backpack': 'рюкзак',
            'umbrella': 'зонт',
            'handbag': 'сумка',
            'tie': 'галстук',
            'suitcase': 'чемодан',
            'frisbee': 'фрисби',
            'skis': 'лыжи',
            'snowboard': 'сноуборд',
            'sports ball': 'мяч',
            'kite': 'воздушный змей',
            'baseball bat': 'бейсбольная бита',
            'baseball glove': 'бейсбольная перчатка',
            'skateboard': 'скейтборд',
            'surfboard': 'доска для серфинга',
            'tennis racket': 'теннисная ракетка',
            'bottle': 'бутылка',
            'wine glass': 'бокал',
            'cup': 'чашка',
            'fork': 'вилка',
            'knife': 'нож',
            'spoon': 'ложка',
            'bowl': 'миска',
            'banana': 'банан',
            'apple': 'яблоко',
            'sandwich': 'сэндвич',
            'orange': 'апельсин',
            'broccoli': 'брокколи',
            'carrot': 'морковь',
            'hot dog': 'хот-дог',
            'pizza': 'пицца',
            'donut': 'пончик',
            'cake': 'торт',
            'chair': 'стул',
            'couch': 'диван',
            'potted plant': 'растение в горшке',
            'bed': 'кровать',
            'dining table': 'обеденный стол',
            'toilet': 'туалет',
            'tv': 'телевизор',
            'laptop': 'ноутбук',
            'mouse': 'мышь',
            'remote': 'пульт',
            'keyboard': 'клавиатура',
            'cell phone': 'телефон',
            'microwave': 'микроволновка',
            'oven': 'духовка',
            'toaster': 'тостер',
            'sink': 'раковина',
            'refrigerator': 'холодильник',
            'book': 'книга',
            'clock': 'часы',
            'vase': 'ваза',
            'scissors': 'ножницы',
            'teddy bear': 'плюшевый мишка',
            'hair drier': 'фен',
            'toothbrush': 'зубная щетка'
        }
    
    def preprocess_image(self, image_bytes: bytes) -> Tuple[Image.Image, float]:
        """
        Предобработка изображения для оптимальной работы с YOLO
        
        Выполняет следующие операции:
        1. Загрузка изображения 
        2. Конвертация в RGB формат (если необходимо)
        3. Масштабирование для ускорения обработки
        
        Args:
            image_bytes: изображения в любом поддерживаемом PIL формате
            
        Returns:
            Tuple[Image.Image, float]: Обработанное изображение и коэффициент масштабирования
        """
        # Загружаем изображение 
        image = Image.open(io.BytesIO(image_bytes))
        
        # Конвертируем в RGB если изображение в другом формате (RGBA, L, P и т.д.)
        # т.к. YOLO работает только с RGB изображениями
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Вычисляем коэффициент масштабирования для уменьшения размера
        # Это ускоряет обработку без значительной потери качества детекции
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > self.max_image_size:
            # Пропорциональное уменьшение изображения
            scale_factor = self.max_image_size / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            # Используем LANCZOS для ресайза (компромисс между скоростью и качеством)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            scale_factor = 1.0
        
        return image, scale_factor
    
    def detect_objects(self, image_bytes: bytes) -> Tuple[List[str], Dict[str, int], bytes]:
        """
        Основная функция детекции объектов на изображении
        
        Процесс работы:
        1. Предобработка изображения
        2. Запуск YOLO модели для детекции
        3. Фильтрация результатов по размеру и уверенности
        4. Перевод названий классов на русский
        5. Подсчет количества объектов каждого класса
        6. Отрисовка ограничивающих рамок
        
        Args:
            image_bytes:  изображение в любом поддерживаемом формате
            
        Returns:
            Tuple[List[str], Dict[str, int], bytes]: 
            - Список уникальных классов объектов (на русском языке)
            - Словарь с количеством найденных объектов каждого класса
            - Изображения с нарисованными ограничивающими рамками
        """
        try:
            # Предобработка изображения (масштабирование и конвертация в RGB)
            image, scale_factor = self.preprocess_image(image_bytes)
            
            # Конвертируем PIL Image в numpy array, который требуется для YOLO
            image_np = np.array(image)
            
            # Запускаем детекцию объектов с заданным порогом уверенности
            logger.info("Выполняю детекцию объектов...")
            results = self.model(image_np, conf=self.confidence_threshold, verbose=False)
            
            # Обрабатываем результаты детекции
            detected_classes = []  # Список уникальных классов
            class_counts = {}      # Счетчик объектов по классам
            
            # Проверяем, что модель нашла объекты
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes  # Получаем все найденные ограничивающие рамки
                
                ## Обрабатываем каждый найденный объект
                for i in range(len(boxes)):
                    # Извлекаем данные об объекте из тензоров PyTorch
                    box = boxes.xyxy[i].cpu().numpy()  # Координаты рамки [x1, y1, x2, y2]
                    class_id = int(boxes.cls[i].cpu().numpy())  # ID класса объекта
                    confidence = float(boxes.conf[i].cpu().numpy())  # Уверенность модели
                    
                    # Фильтруем слишком маленькие объекты (возможные ложные срабатывания)
                    width = box[2] - box[0]   # Ширина объекта
                    height = box[3] - box[1]  # Высота объекта
                    area = width * height     # Площадь объекта
                    
                    if area < self.min_object_area:
                        continue  # Пропускаем слишком маленькие объекты
                    
                    # Получаем английское название класса из модели YOLO
                    class_name = self.model.names[class_id]
                    
                    # Переводим на русский язык, если перевод доступен
                    russian_name = self.class_translations.get(class_name, class_name)
                    
                    # Добавляем класс в список уникальных классов
                    if russian_name not in detected_classes:
                        detected_classes.append(russian_name)
                    
                    # Увеличиваем счетчик объектов данного класса
                    if russian_name in class_counts:
                        class_counts[russian_name] += 1
                    else:
                        class_counts[russian_name] = 1
            
            # Создаем изображение с нарисованными ограничивающими рамками
            annotated_image = self.draw_boxes(image, results[0] if len(results) > 0 else None)
            
            # Конвертируем PIL Image обратно в изображение .jpeg для отправки пользователю
            img_buffer = io.BytesIO()
            annotated_image.save(img_buffer, format='JPEG', quality=85)  # Сжимаем для экономии трафика
            annotated_image_bytes = img_buffer.getvalue()
            
            logger.info(f"Детекция завершена. Найдено классов: {len(detected_classes)}")
            
            return detected_classes, class_counts, annotated_image_bytes
            
        except Exception as e:
            logger.error(f"Ошибка при детекции объектов: {e}")
            raise
    
    def draw_boxes(self, image: Image.Image, results) -> Image.Image:
        """
        Отрисовка ограничивающих рамок вокруг найденных объектов
        
        Функция рисует:
        - Цветные прямоугольные рамки вокруг объектов
        - Подписи с названием класса и уверенностью модели
        - Цветной фон под текстом для лучшей читаемости
        
        Args:
            image: Исходное изображение PIL
            results: Результаты детекции от YOLO модели
            
        Returns:
            Image.Image: Изображение с нарисованными рамками и подписями
        """
        # Создаем копию изображения для рисования (не изменяем оригинал)
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Пытаемся загрузить системный шрифт для красивых подписей
        try:
            # Попытка загрузить Arial (Windows)
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                # Попытка загрузить Arial (macOS)
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                # Используем стандартный шрифт PIL если системный недоступен
                font = ImageFont.load_default()
        
        # Если нет результатов детекции, возвращаем исходное изображение
        if results is None or results.boxes is None:
            return annotated_image
        
        boxes = results.boxes
        
        # Палитра ярких и контрастных цветов для визуального различения разных классов объектов
        colors = [
            '#FF0000',  # Красный
            '#00FF00',  # Зеленый  
            '#0000FF',  # Синий
            '#FFFF00',  # Желтый
            '#FF00FF',  # Магента
            '#00FFFF',  # Циан
            '#FFA500',  # Оранжевый
            '#800080',  # Фиолетовый
            '#008000',  # Темно-зеленый
            '#FF1493'   # Темно-розовый
        ]
        
        ## Обрабатываем каждый найденный объект
        for i in range(len(boxes)):
            # Извлекаем данные об объекте
            box = boxes.xyxy[i].cpu().numpy()  # Координаты ограничивающей рамки
            class_id = int(boxes.cls[i].cpu().numpy())  # ID класса
            confidence = float(boxes.conf[i].cpu().numpy())  # Уверенность модели
            
            # Фильтруем маленькие объекты (как выше в функции detect_objects)
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            
            if area < self.min_object_area:
                continue  # Пропускаем слишком маленькие объекты
            
            # Получаем название класса и переводим на русский
            class_name = self.model.names[class_id]
            russian_name = self.class_translations.get(class_name, class_name)
            
            # Выбираем цвет для класса (циклически по палитре выше)
            color = colors[class_id % len(colors)]
            
            # Рисуем прямоугольную рамку вокруг объекта
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])],  # Координаты левого верхнего и правого нижнего углов
                outline=color,  # Цвет контура
                width=3         # Толщина линии
            )
            
            # Создаем подпись с названием класса и уверенностью
            label = f"{russian_name} {confidence:.2f}"
            
            # Вычисляем размеры текста для создания фона
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]   # Ширина текста
            text_height = bbox[3] - bbox[1]  # Высота текста
            
            # Рисуем цветной прямоугольник-фон для текста (для лучшей читаемости)
            draw.rectangle(
                [(box[0], box[1] - text_height - 4), (box[0] + text_width + 4, box[1])],
                fill=color  # Заливаем тем же цветом, что и рамка
            )
            
            # Рисуем белый текст поверх цветного фона
            draw.text(
                (box[0] + 2, box[1] - text_height - 2),  # Позиция с небольшим отступом
                label,        # Текст подписи
                fill='white', # Белый цвет для контраста
                font=font     # Загруженный шрифт
            )
        
        return annotated_image
    
    def segment_objects(self, image_bytes: bytes, selected_classes: List[str]) -> Dict[str, bytes]:
        """
        Сегментация объектов для выбранных пользователем классов
        
        Процесс работы:
        1. Предобработка изображения
        2. Запуск YOLO модели сегментации
        3. Фильтрация результатов по выбранным классам
        4. Группировка масок по классам объектов
        5. Создание цветных изображений с масками для каждого класса
        
        Args:
            image_bytes: Исходное изображение
            selected_classes: Список классов для сегментации (названия на русском языке)
            
        Returns:
            Dict[str, bytes]: Словарь где ключ - название класса, значение - PNG изображение с маской
        """
        try:
            # Предобработка изображения (аналогично детекции)
            image, scale_factor = self.preprocess_image(image_bytes)
            
            # Конвертируем PIL Image в numpy array для модели сегментации
            image_np = np.array(image)
            
            # Запускаем модель сегментации YOLO (возвращает маски вместо просто рамок)
            logger.info("Выполняю сегментацию объектов...")
            results = self.seg_model(image_np, conf=self.confidence_threshold, verbose=False)
            
            # Словарь для хранения результатов сегментации по классам
            segmentation_results = {}
            
            # Проверяем, что модель нашла объекты и создала маски
            if len(results) > 0 and results[0].masks is not None:
                # Группируем маски сегментации по классам объектов
                class_masks = {}  # {класс: [список_масок]}
                boxes = results[0].boxes   # Ограничивающие рамки
                masks = results[0].masks   # Маски сегментации
                
                # Обрабатываем каждый найденный объект
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].cpu().numpy())      # ID класса
                    confidence = float(boxes.conf[i].cpu().numpy()) # Уверенность
                    
                    # Получаем название класса и переводим на русский
                    class_name = self.seg_model.names[class_id]
                    russian_name = self.class_translations.get(class_name, class_name)
                    
                    # Проверяем, выбрал ли пользователь этот класс для сегментации
                    if russian_name in selected_classes:
                        # Инициализируем список масок для класса, если его еще нет
                        if russian_name not in class_masks:
                            class_masks[russian_name] = []
                        
                        # Извлекаем маску сегментации из тензора PyTorch
                        mask = masks.data[i].cpu().numpy()
                        
                        # Сохраняем маску с дополнительной информацией
                        class_masks[russian_name].append({
                            'mask': mask,           # Бинарная маска объекта
                            'confidence': confidence, # Уверенность модели
                            'class_name': class_name  # Оригинальное английское название
                        })
                
                # Создаем итоговые изображения с масками для каждого класса
                for class_name, masks_data in class_masks.items():
                    # Создаем изображение с цветными масками для класса
                    class_image = self.create_segmentation_image(image, masks_data, class_name)
                    
                    if class_image:
                        # Конвертируем PIL Image в байты PNG (без потери качества)
                        img_buffer = io.BytesIO()
                        class_image.save(img_buffer, format='PNG')
                        segmentation_results[class_name] = img_buffer.getvalue()
            
            logger.info(f"Сегментация завершена. Создано изображений: {len(segmentation_results)}")
            return segmentation_results
            
        except Exception as e:
            logger.error(f"Ошибка при сегментации объектов: {e}")
            return {}
    
    def create_segmentation_image(self, image: Image.Image, masks_data: List[Dict], class_name: str) -> Image.Image:
        """
        Создание итогового изображения с цветными масками сегментации для конкретного класса
        
        Функция выполняет:
        1. Объединение всех масок класса в одну общую маску
        2. Создание цветной полупрозрачной маски
        3. Наложение маски на исходное изображение
        4. Отрисовку контуров объектов
        5. Добавление текстовой подписи с количеством найденных объектов
        
        Args:
            image: Исходное изображение PIL
            masks_data: Список словарей с данными масок для одного класса
            class_name: Название класса на русском языке
            
        Returns:
            Image.Image: Изображение с наложенными цветными масками или None при ошибке
        """
        try:
            # Создаем копию изображения с поддержкой прозрачности (RGBA)
            result_image = image.copy().convert('RGBA')
            width, height = image.size
            
            # Предопределенные цвета для популярных классов объектов
            # Формат: (R, G, B, Alpha) где Alpha = прозрачность (120 = полупрозрачный)
            class_colors = {
                'человек': (255, 0, 0, 120),        # Красный - для людей
                'автомобиль': (0, 255, 0, 120),     # Зеленый - для машин
                'велосипед': (0, 0, 255, 120),      # Синий - для велосипедов
                'собака': (255, 255, 0, 120),       # Желтый - для собак
                'кот': (255, 0, 255, 120),          # Магента - для кошек
                'стул': (0, 255, 255, 120),         # Циан - для мебели
                'бутылка': (255, 128, 0, 120),      # Оранжевый - для бутылок
                'чашка': (128, 0, 255, 120),        # Фиолетовый - для посуды
                'книга': (0, 128, 255, 120),        # Голубой - для книг
                'телефон': (255, 0, 128, 120),      # Розовый - для телефонов
                'телевизор': (128, 255, 0, 120),    # Лайм - для техники
                'ноутбук': (255, 128, 128, 120),    # Светло-красный - для компьютеров
                'мышь': (128, 128, 255, 120),       # Светло-синий - для периферии
                'клавиатура': (255, 255, 128, 120), # Светло-желтый - для клавиатур
                'часы': (128, 255, 255, 120),       # Светло-циан - для часов
            }
            
            # Получаем цвет для класса или используем БЕЛЫЙ по умолчанию
            color = class_colors.get(class_name, (255, 255, 255, 120))
            
            # Создаем общую бинарную маску, объединяющую все объекты данного класса
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Объединяем все маски объектов одного класса
            for mask_data in masks_data:
                mask = mask_data['mask']
                
                # Приводим маску к размеру изображения (если размеры не совпадают)
                if mask.shape != (height, width):
                    mask_resized = cv2.resize(mask.astype(np.uint8), (width, height))
                else:
                    mask_resized = mask.astype(np.uint8)
                
                # Объединяем маски через операцию максимума (логическое ИЛИ)
                combined_mask = np.maximum(combined_mask, mask_resized)
            
            # Создаем цветную RGBA маску из бинарной маски
            colored_mask = np.zeros((height, width, 4), dtype=np.uint8)
            colored_mask[combined_mask > 0] = color  # Закрашиваем области объектов выбранным цветом
            
            # Конвертируем numpy array в PIL Image с поддержкой прозрачности
            mask_image = Image.fromarray(colored_mask, 'RGBA')
            
            # Накладываем полупрозрачную цветную маску на исходное изображение
            result_image = Image.alpha_composite(result_image, mask_image)
            
            # Находим контуры объектов для отрисовки границ
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Конвертируем PIL Image обратно в numpy array для работы с OpenCV
            result_np = np.array(result_image)
            
            # Рисуем контуры объектов (границы) поверх цветной маски
            cv2.drawContours(result_np, contours, -1, color[:3], 2)  # Используем RGB без альфа-канала
            
            # Конвертируем обратно в PIL Image и убираем альфа-канал для финального изображения
            result_image = Image.fromarray(result_np, 'RGBA').convert('RGB')
            
            # Добавляем текстовую подпись с информацией о найденных объектах
            draw = ImageDraw.Draw(result_image)
            
            # Пытаемся загрузить системный шрифт большего размера для подписи
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Формируем текст с количеством найденных объектов
            object_count = len(masks_data)
            text = f"{class_name}: {object_count} объект(ов)"
            
            # Рисуем текст с эффектом тени для лучшей читаемости
            draw.text((12, 12), text, fill='black', font=font)  # Черная тень (смещение на 2 пикселя)
            draw.text((10, 10), text, fill='white', font=font)  # Белый основной текст
            
            return result_image
        
        ## если ошибка
        except Exception as e:
            logger.error(f"Ошибка создания изображения сегментации для {class_name}: {e}")
            return None