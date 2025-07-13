import os
import argparse
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translate_files_in_folder(input_dir, output_dir, mt_model_name):
    """
    Переводит все .txt файлы из input_dir на русский язык и сохраняет их в output_dir.

    Args:
        input_dir (str): Путь к входной папке с файлами для перевода.
        output_dir (str): Путь к выходной папке, куда будут сохранены переведенные файлы.
        mt_model_name (str): Название модели для перевода (например, "facebook/wmt19-en-ru").
    """
    print(f"Начало процесса перевода из '{input_dir}' в '{output_dir}'")

    # 1. Проверка существования входной папки
    if not os.path.isdir(input_dir):
        print(f"Ошибка: Входная папка '{input_dir}' не найдена.")
        return

    # 2. Создание выходной папки, если она не существует
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Выходная папка '{output_dir}' проверена/создана.")
    except OSError as e:
        print(f"Ошибка при создании/проверке выходной папки '{output_dir}': {e}")
        return

    # 3. Загрузка модели и токенизатора (один раз)
    assert mt_model_name in ["Helsinki-NLP/opus-mt-en-ru", "facebook/wmt19-en-ru"]
    mt_tokenizer = None
    mt_model = None

    print(f"Загрузка модели перевода '{mt_model_name}' (это может занять некоторое время)...")
    try:
        if mt_model_name == "facebook/wmt19-en-ru":
            mt_tokenizer = FSMTTokenizer.from_pretrained(mt_model_name)
            mt_model = FSMTForConditionalGeneration.from_pretrained(mt_model_name)
            print(f"Модель перевода {mt_model_name} успешно загружена.")
        elif mt_model_name == "Helsinki-NLP/opus-mt-en-ru":
            mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_name)
            mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name)
            print(f"Модель перевода {mt_model_name} успешно загружена.")
        else:
            raise ValueError(f"Неизвестное имя модели: {mt_model_name}")
    except Exception as e:
        print(f"Ошибка при загрузке модели {mt_model_name}: {e}")
        print("Пожалуйста, убедитесь, что у вас есть подключение к интернету и библиотека 'transformers' установлена корректно.")
        return

    # 4. Проход по файлам во входной папке
    files_translated = 0
    files_skipped = 0

    # Находим поддиректории для каждого исходного аудио
    # (например, 'Bill_Gates', 'Cameron_Russell')
    subdirectories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirectories:
        print(f"Error: No subdirectories found in '{input_dir}'. Expected format is 'input_dir/audio_name/chunks.wav'.")
        return

    # Обработка каждой поддиректории
    for subdir_name in subdirectories:
        os.makedirs(os.path.join(output_dir, subdir_name), exist_ok=True)
        print(f"\nProcessing audio: {subdir_name}")

        full_subdir_path = os.path.join(input_dir, subdir_name)  # transcriptions/Bill_Gates

        for filename in os.listdir(full_subdir_path):
            if filename.endswith(".txt"):
                input_filepath = os.path.join(full_subdir_path, filename)
                output_filepath = os.path.join(output_dir, subdir_name, filename)

                print(f"\nОбработка файла: '{filename}'")

                try:
                    # Чтение содержимого файла
                    with open(input_filepath, "r", encoding="utf-8") as f:
                        english_text = f.read()

                    # Проверка на пустой файл или файл только с пробелами
                    if not english_text.strip():
                        print(f"  Файл '{filename}' пуст или содержит только пробелы. Перевод пропущен.")
                        russian_text = ""  # Записываем пустую строку в выходной файл
                        files_skipped += 1
                    else:
                        # Выполнение перевода
                        print("  Выполнение перевода...")
                        input_ids = mt_tokenizer.encode(english_text, return_tensors="pt")
                        outputs = mt_model.generate(input_ids)
                        russian_text = mt_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        print("  Перевод завершен.")
                        files_translated += 1

                    # Запись переведенного текста в новый файл
                    with open(output_filepath, "w", encoding="utf-8") as f:
                        f.write(russian_text)
                    print(f"  Переведенный текст сохранен в: '{output_filepath}'")

                except FileNotFoundError:
                    print(f"  Предупреждение: Файл '{input_filepath}' не найден, пропущен.")
                    files_skipped += 1
                except Exception as e:
                    print(f"  Ошибка при обработке файла '{filename}': {e}")
                    files_skipped += 1

    print("\n--- Процесс перевода завершен ---")
    print(f"Переведено файлов: {files_translated}")
    print(f"Пропущено файлов (пустые или с ошибками): {files_skipped}")

    print(f"Перевод файлов в {input_dir} (не в поддиректориях):")
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)

            print(f"\nОбработка файла: '{filename}'")

            try:
                # Чтение содержимого файла
                with open(input_filepath, "r", encoding="utf-8") as f:
                    english_text = f.read()

                # Проверка на пустой файл или файл только с пробелами
                if not english_text.strip():
                    print(f"  Файл '{filename}' пуст или содержит только пробелы. Перевод пропущен.")
                    russian_text = ""  # Записываем пустую строку в выходной файл
                    files_skipped += 1
                else:
                    # Выполнение перевода
                    print("  Выполнение перевода...")
                    input_ids = mt_tokenizer.encode(english_text, return_tensors="pt")
                    outputs = mt_model.generate(input_ids)
                    russian_text = mt_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print("  Перевод завершен.")
                    files_translated += 1

                # Запись переведенного текста в новый файл
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(russian_text)
                print(f"  Переведенный текст сохранен в: '{output_filepath}'")

            except FileNotFoundError:
                print(f"  Предупреждение: Файл '{input_filepath}' не найден, пропущен.")
                files_skipped += 1
            except Exception as e:
                print(f"  Ошибка при обработке файла '{filename}': {e}")
                files_skipped += 1

    print(f"Все переведенные файлы находятся в: '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Переводит все .txt файлы из одной папки на русский язык и сохраняет их в другую папку."
    )
    parser.add_argument(
        "input_folder",
        nargs='?',  # Делает аргумент необязательным
        default="transcriptions",
        help="Путь к входной папке с файлами для перевода (по умолчанию: transcriptions)"
    )
    parser.add_argument(
        "output_folder",
        nargs='?',
        default="translations",
        help="Путь к выходной папке для сохранения переведенных файлов (по умолчанию: translations)"
    )
    parser.add_argument(
        "mt_model_name",
        nargs='?',
        default="Helsinki-NLP/opus-mt-en-ru",
        help="Название модели для перевода (например, 'Helsinki-NLP/opus-mt-en-ru' или 'facebook/wmt19-en-ru')"
    )

    args = parser.parse_args()

    translate_files_in_folder(args.input_folder, args.output_folder, args.mt_model_name)

# Usage example:
# python translate_en_ru.py transcriptions translations
