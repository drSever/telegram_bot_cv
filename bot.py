### Импорты библиотек

# для логирования и асинхронного программирования
import logging
import asyncio
from typing import Dict, List, Optional

# для работы с Telegram Bot API
from telegram import Update, InputMediaPhoto
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# для работы с изображениями
import io
from PIL import Image

# Локальные модули проекта
from config import config  # Конфигурация бота (токены, сообщения)
from detection import YOLOObjectDetector  # Класс для детекции объектов

### Настройка логирования для отслеживания работы бота
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

### Telegram-Bot
class ObjectDetectionBot:
    """
    Telegram бот для детекции и сегментации объектов
    
    Основные возможности:
    - Детекция объектов на изображениях с помощью YOLO
    - Сегментация выбранных пользователем объектов
    - Поддержка русского и английского языков для названий классов
    - Интеллектуальное сопоставление пользовательского ввода
    """
    
    def __init__(self):
        """
        Инициализация бота
        
        Создает:
        - Словарь для хранения сессий пользователей
        - Экземпляр детектора YOLO
        """
        # Словарь для хранения состояний пользователей
        # Ключ: user_id (int), Значение: dict с данными сессии
        self.user_sessions: Dict[int, dict] = {}
        
        # Инициализируем детектор объектов с обработкой ошибок
        try:
            self.detector = YOLOObjectDetector()
            logger.info("Детектор объектов успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации детектора: {e}")
            self.detector = None  # Если модель не загрузилась, бот продолжит работу
        

    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /start
        
        Инициализирует новую сессию пользователя и отправляет приветственное сообщение
        """
        user_id = update.effective_user.id
        
        # Создаем новую сессию пользователя с начальным состоянием
        self.user_sessions[user_id] = {'state': 'waiting_photo'}
        
        # Отправляем приветственное сообщение с HTML-разметкой
        await update.message.reply_text(
            config.START_MESSAGE,
            parse_mode=ParseMode.HTML
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /help
        
        Отправляет пользователю справочную информацию о возможностях бота
        """
        await update.message.reply_text(
            config.HELP_MESSAGE,
            parse_mode=ParseMode.HTML
        )
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик получения фотографии от пользователя
        
        Выполняет:
        1. Проверку доступности детектора
        2. Загрузку и обработку изображения
        3. Детекцию объектов с помощью YOLO
        4. Отправку результатов пользователю
        """
        user_id = update.effective_user.id
        
        # Проверяем, что детектор успешно инициализирован
        if self.detector is None:
            await update.message.reply_text("❌ Детектор объектов недоступен. Попробуйте позже.")
            return
        
        try:
            # Уведомляем пользователя о начале обработки
            status_message = await update.message.reply_text(config.PHOTO_RECEIVED)
            
            # Получаем файл фотографии (берем самое большое разрешение [-1])
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            
            # Выполняем детекцию объектов в отдельном потоке
            # чтобы не блокировать основной поток бота
            detected_classes, class_counts, annotated_image_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self.detector.detect_objects, bytes(photo_bytes)
            )
            
            # Проверяем, найдены ли объекты на изображении
            if not detected_classes:
                await update.message.reply_text(config.ERROR_NO_OBJECTS)
                return
            
            # Сохраняем все данные в сессии пользователя для дальнейшего использования
            self.user_sessions[user_id] = {
                'state': 'waiting_class_selection',  # Переходим к этапу выбора классов
                'image_bytes': photo_bytes,          # Оригинальное изображение
                'detected_classes': detected_classes, # Список найденных классов
                'class_counts': class_counts,        # Количество объектов каждого класса
                'annotated_image_bytes': annotated_image_bytes  # Изображение с рамками
            }
            
            # Отправляем пользователю изображение с выделенными объектами
            await update.message.reply_photo(
                photo=io.BytesIO(annotated_image_bytes),
                caption="🔍 Найденные объекты отмечены рамками"
            )
            
            # Формируем список найденных классов с количеством объектов
            classes_text = []
            for cls in detected_classes:
                count = class_counts.get(cls, 0)
                classes_text.append(f"• {cls} ({count} шт.)")
            
            classes_list = '\n'.join(classes_text)
            
            # Формируем подсказку с английскими названиями для удобства ввода
            english_classes = []
            for rus_class in detected_classes:
                # Ищем английское название в словаре переводов
                eng_class = None
                for eng, rus in self.detector.class_translations.items():
                    if rus == rus_class:
                        eng_class = eng
                        break
                if eng_class:
                    english_classes.append(eng_class)
                else:
                    english_classes.append(rus_class)  # Если перевода нет, используем оригинал
            
            # Показываем первые 5 классов в подсказке, чтобы не перегружать сообщение
            hint_text = ' '.join(english_classes[:5])
            if len(english_classes) > 5:
                hint_text += " ..."
            
            # Формируем итоговое сообщение с результатами и подсказкой
            response_text = config.DETECTION_COMPLETE.format(classes_list)
            response_text += f"\n\n💡 Пример: {hint_text}"
            
            await update.message.reply_text(response_text)
            
        except Exception as e:
            # Логируем ошибку для отладки и отправляем пользователю понятное сообщение
            logger.error(f"Ошибка при обработке фото: {e}")
            await update.message.reply_text(config.ERROR_PROCESSING)
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик текстовых сообщений от пользователя
        
        Маршрутизирует сообщения в зависимости от текущего состояния пользователя:
        - waiting_photo: просит отправить фото
        - waiting_class_selection: обрабатывает выбор классов
        - другие состояния: предлагает начать заново
        """
        user_id = update.effective_user.id
        text = update.message.text.strip().lower()  # Нормализуем текст
        
        # Проверяем, есть ли активная сессия пользователя
        if user_id not in self.user_sessions:
            await update.message.reply_text(
                "Начни с команды /start и отправь фотографию для анализа."
            )
            return
        
        session = self.user_sessions[user_id]
        
        # Маршрутизация по состояниям
        if session['state'] == 'waiting_photo':
            # Пользователь отправил текст, но нужно фото
            await update.message.reply_text(config.ERROR_NO_PHOTO)
            return
        
        elif session['state'] == 'waiting_class_selection':
            # Обрабатываем выбор классов для сегментации
            await self.handle_class_selection(update, context, text)
        
        else:
            # Неизвестное состояние - предлагаем начать заново
            await update.message.reply_text(
                "Отправь новую фотографию для анализа или используй /start"
            )
    
    def parse_user_classes(self, text: str) -> List[str]:
        """
        Устойчивый парсер пользовательского ввода классов
        
        Обрабатывает различные форматы ввода: через пробелы, запятые, точку с запятой, 
        с лишними символами
        
        Args:
            text: Текст от пользователя
            
        Returns:
            List[str]: Список очищенных названий классов
        """
        import re
        
        # Убираем все символы кроме букв, цифр, пробелов и основных разделителей
        # Сохраняем русские буквы (а-яё) для поддержки русских названий
        text = re.sub(r'[^\w\s,;а-яё]', ' ', text.lower())
        
        # Разделяем текст по различным разделителям (пробелы, запятые, точки с запятой)
        classes = re.split(r'[,;\s]+', text)
        
        # Фильтруем результат: убираем пустые строки и слишком короткие слова
        cleaned_classes = []
        for cls in classes:
            cls = cls.strip()
            if len(cls) >= 2:  # Минимум 2 символа для валидного названия класса
                cleaned_classes.append(cls)
        
        return cleaned_classes
    
    def match_user_classes_to_detected(self, user_classes: List[str], detected_classes: List[str]) -> tuple:
        """
        Сопоставление пользовательских классов с найденными на изображении
        
        3 уровня поиска:
        1. Точное совпадение с русскими названиями
        2. Совпадение с английскими названиями (через словарь переводов)
        3. Нечеткое совпадение для исправления опечаток
        
        Args:
            user_classes: Классы, введенные пользователем
            detected_classes: Классы, найденные на изображении (на русском языке)
            
        Returns:
            tuple: (найденные_классы, не_найденные_классы, предложения_исправлений)
        """
        valid_classes = []      # Успешно сопоставленные классы
        invalid_classes = []    # Классы, которые не удалось найти
        suggestions = []        # Предложения исправлений для пользователя
        
        for user_class in user_classes:
            found = False
            
            # Уровень 1: Прямое совпадение с русскими названиями
            for detected_class in detected_classes:
                if user_class == detected_class.lower():
                    if detected_class not in valid_classes:  # Избегаем дублирования
                        valid_classes.append(detected_class)
                    found = True
                    break
            
            if found:
                continue
            
            # Уровень 2: Совпадение с английскими названиями через словарь переводов
            for eng_name, rus_name in self.detector.class_translations.items():
                if user_class == eng_name.lower() and rus_name in detected_classes:
                    if rus_name not in valid_classes:
                        valid_classes.append(rus_name)
                    found = True
                    break
            
            if found:
                continue
            
            # Уровень 3: Нечеткое совпадение для исправления опечаток
            best_match = None
            best_score = 0
            
            # Проверяем схожесть с русскими названиями
            for detected_class in detected_classes:
                score = self.calculate_similarity(user_class, detected_class.lower())
                if score > best_score and score > 0.6:  # Порог схожести 60%
                    best_score = score
                    best_match = detected_class
            
            # Проверяем схожесть с английскими названиями
            for eng_name, rus_name in self.detector.class_translations.items():
                if rus_name in detected_classes:
                    score = self.calculate_similarity(user_class, eng_name.lower())
                    if score > best_score and score > 0.6:
                        best_score = score
                        best_match = rus_name
            
            # Добавляем результат нечеткого поиска
            if best_match and best_match not in valid_classes:
                valid_classes.append(best_match)
                suggestions.append(f"'{user_class}' → '{best_match}'")
            else:
                invalid_classes.append(user_class)
        
        return valid_classes, invalid_classes, suggestions
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Вычисляет схожесть двух строк для нечеткого поиска
        
        Использует простой алгоритм на основе:
        1. Точного совпадения (1.0)
        2. Вхождения одной строки в другую (0.8)
        3. Доли общих символов (0.0-1.0)
        
        Args:
            str1: Первая строка для сравнения
            str2: Вторая строка для сравнения
            
        Returns:
            float: Коэффициент схожести от 0.0 до 1.0
        """
        # Точное совпадение
        if str1 == str2:
            return 1.0
        
        # Проверяем, содержится ли одна строка в другой
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Вычисляем схожесть на основе общих символов
        common_chars = set(str1) & set(str2)  # Пересечение множеств символов
        total_chars = set(str1) | set(str2)   # Объединение множеств символов
        
        if not total_chars:
            return 0.0
        
        # Возвращаем долю общих символов
        return len(common_chars) / len(total_chars)

    async def handle_class_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
        """
        Обработка выбора классов для сегментации
        
        Выполняет:
        1. Парсинг пользовательского ввода
        2. Сопоставление с найденными классами
        3. Формирование ответа с результатами
        4. Запуск процесса сегментации для валидных классов
        """
        user_id = update.effective_user.id
        session = self.user_sessions[user_id]
        
        # Получаем данные из сессии пользователя
        detected_classes = session['detected_classes']
        class_counts = session['class_counts']
        
        # Парсим и очищаем пользовательский ввод
        user_classes = self.parse_user_classes(text)
        
        # Проверяем, удалось ли распознать хотя бы один класс
        if not user_classes:
            await update.message.reply_text(
                "❌ Не удалось распознать классы. Попробуйте еще раз.\n"
                "Например: car person bicycle"
            )
            return
        
        # Сопоставляем пользовательский ввод с найденными на изображении классами
        valid_classes, invalid_classes, suggestions = self.match_user_classes_to_detected(
            user_classes, detected_classes
        )
        
        # Формируем детальный ответ пользователю о результатах сопоставления
        response_parts = []
        
        # Показываем успешно найденные классы
        if valid_classes:
            selected_text = ', '.join(valid_classes)
            response_parts.append(f"✅ Выбраны классы: {selected_text}")
            
            # Если были исправления, показываем их пользователю
            if suggestions:
                suggestions_text = '\n'.join([f"  • {s}" for s in suggestions])
                response_parts.append(f"🔄 Исправления:\n{suggestions_text}")
        
        # Показываем классы, которые не удалось найти
        if invalid_classes:
            invalid_text = ', '.join(invalid_classes)
            response_parts.append(f"❌ Не найдены: {invalid_text}")
            
            # Формируем список доступных классов для подсказки
            available_classes = []
            for rus_class in detected_classes:
                # Ищем английское название для каждого русского класса
                eng_class = None
                for eng, rus in self.detector.class_translations.items():
                    if rus == rus_class:
                        eng_class = eng
                        break
                if eng_class:
                    available_classes.append(f"{eng_class}")
                else:
                    available_classes.append(rus_class)
            
            response_parts.append(f"📋 Доступные: {', '.join(available_classes)}")
        
        # Если не найдено ни одного валидного класса, отправляем только информационное сообщение
        if not valid_classes:
            await update.message.reply_text('\n\n'.join(response_parts))
            return
        
        # Отправляем ответ с результатами сопоставления
        await update.message.reply_text('\n\n'.join(response_parts))
        
        # Сохраняем выбранные классы в сессии и меняем состояние
        session['selected_classes'] = valid_classes
        session['state'] = 'processing_segmentation'
        
        # Уведомляем о начале процесса сегментации
        await update.message.reply_text(config.SEGMENTATION_START)
        
        # Запускаем процесс сегментации
        await self.perform_segmentation(update, session)
    
    async def perform_segmentation(self, update: Update, session: dict) -> None:
        """
        Выполняет сегментацию объектов для выбранных пользователем классов
        
        Процесс:
        1. Вызывает метод сегментации детектора YOLO
        2. Отправляет результаты по каждому классу отдельно
        3. Формирует итоговую статистику
        4. Сбрасывает состояние пользователя для новой сессии
        """
        try:
            # Проверяем доступность детектора
            if self.detector is None:
                await update.message.reply_text("❌ Детектор недоступен")
                return
            
            # Извлекаем необходимые данные из сессии
            selected_classes = session['selected_classes']
            image_bytes = session['image_bytes']
            class_counts = session['class_counts']
            
            # Выполняем сегментацию в отдельном потоке для каждого выбранного класса
            # Это предотвращает блокировку основного потока бота
            segmentation_results = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.detector.segment_objects,
                image_bytes,
                selected_classes
            )
            
            # Обрабатываем результаты сегментации
            successful_classes = []  # Классы, для которых сегментация прошла успешно
            failed_classes = []      # Классы, для которых сегментация не удалась
            
            for class_name in selected_classes:
                if class_name in segmentation_results:
                    # Отправляем изображение с выделенными объектами данного класса
                    await update.message.reply_photo(
                        photo=io.BytesIO(segmentation_results[class_name]),
                        caption=f"🎯 Сегментация: {class_name}"
                    )
                    successful_classes.append(class_name)
                else:
                    # Сегментация не удалась (возможно, объекты слишком мелкие или нечеткие)
                    failed_classes.append(class_name)
            
            # Отправляем итоговую сводку результатов
            await self.send_final_summary(update, successful_classes, failed_classes, class_counts)
            
            # Возвращаем пользователя в начальное состояние для обработки новых фото
            session['state'] = 'waiting_photo'
            
        except Exception as e:
            # Логируем ошибку и уведомляем пользователя
            logger.error(f"Ошибка при сегментации: {e}")
            await update.message.reply_text(config.ERROR_PROCESSING)
    
    async def send_final_summary(self, update: Update, successful_classes: List[str], 
                                failed_classes: List[str], class_counts: Dict[str, int]) -> None:
        """
        Отправляет финальную сводку результатов сегментации
        
        Включает:
        - Список успешно обработанных классов с количеством объектов
        - Список неудачных попыток с подсказками
        - Общую статистику обработанных объектов
        - Приглашение к дальнейшему использованию
        """
        
        summary_parts = []
        
        # Формируем отчет об успешных результатах
        if successful_classes:
            summary_parts.append("📊 Результаты сегментации:")
            for class_name in successful_classes:
                count = class_counts.get(class_name, 0)
                summary_parts.append(f"• {class_name}: {count} шт.")
        
        # Формируем отчет о неудачных попытках с подсказками
        if failed_classes:
            summary_parts.append("\n❌ Не удалось сегментировать:")
            for class_name in failed_classes:
                # Ищем английское название для подсказки пользователю
                eng_name = None
                for eng, rus in self.detector.class_translations.items():
                    if rus == class_name:
                        eng_name = eng
                        break
                
                if eng_name:
                    summary_parts.append(f"• {class_name} (попробуйте '{eng_name}')")
                else:
                    summary_parts.append(f"• {class_name}")
        
        # Подсчитываем общее количество успешно обработанных объектов
        total_objects = sum(class_counts.get(cls, 0) for cls in successful_classes)
        if total_objects > 0:
            summary_parts.append(f"\n🎉 Всего обработано объектов: {total_objects}")
        
        # Добавляем приглашение к дальнейшему использованию
        summary_parts.append("\n💡 Отправьте новое фото для анализа!")
        
        # Отправляем сформированную сводку пользователю
        await update.message.reply_text('\n'.join(summary_parts))
    
    async def handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Централизованный обработчик ошибок бота
        
        Выполняет:
        - Логирование ошибки для отладки
        - Отправку пользователю понятного сообщения об ошибке
        - Предотвращение краха бота при неожиданных ошибках
        """
        logger.error(f"Ошибка: {context.error}")
        
        # Отправляем пользователю сообщение об ошибке, если возможно
        if update.message:
            await update.message.reply_text(config.ERROR_PROCESSING)

def main():
    """
    Главная функция для запуска Telegram бота
    """
    # Проверяем наличие токена бота в конфигурации
    if not config.BOT_TOKEN:
        logger.error("BOT_TOKEN не установлен! Установите переменную окружения BOT_TOKEN")
        return
    
    # Создаем основное приложение бота с токеном
    application = Application.builder().token(config.BOT_TOKEN).build()
    
    # Создаем экземпляр нашего бота
    bot = ObjectDetectionBot()
    
    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(CommandHandler("help", bot.help_command))
    
    # Регистрируем обработчики сообщений
    application.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))  # Для фотографий
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))  # Для текста (кроме команд)
    
    # Регистрируем глобальный обработчик ошибок
    application.add_error_handler(bot.handle_error)
    
    logger.info("Бот запущен!")
    
    # Запускаем бота в режиме polling (постоянное получение обновлений от Telegram)
    # allowed_updates=Update.ALL_TYPES позволяет получать все типы обновлений
    application.run_polling(allowed_updates=Update.ALL_TYPES)

### Запуск
if __name__ == '__main__':
    main()