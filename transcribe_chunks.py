import whisper
import os
import glob
import argparse
from tqdm import tqdm
import torch


def get_chunk_number(filename: str) -> int:
    """Извлекает номер чанка из имени файла для правильной сортировки."""
    try:
        # Имя файла выглядит как '...-chunk-123.wav'
        base = os.path.basename(filename)
        num_str = base.split('-')[-1].replace('.wav', '')
        return int(num_str)
    except (IndexError, ValueError):
        # Если формат имени файла неожиданный, возвращаем 0, чтобы он был в начале
        print(f"Warning: Could not parse chunk number from '{filename}'")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio chunks using OpenAI's Whisper.")

    parser.add_argument('-i', '--input-dir', required=True,
                        help='Directory containing subdirectories of audio chunks (e.g., splited_audio_robust).')
    parser.add_argument('-o', '--output-dir', default='transcriptions',
                        help='Directory to save the final transcription files (default: transcriptions).')
    parser.add_argument('-m', '--model-size', default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                        help='Whisper model size (default: base).')
    parser.add_argument('-l', '--language', default='en',
                        help='Language of the audio files (default: en for English).')

    args = parser.parse_args()

    # Проверка наличия GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: Running on CPU. Transcription will be much slower. Consider installing PyTorch with CUDA support if you have an NVIDIA GPU.")

    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)

    # Загрузка модели Whisper
    print(f"Loading Whisper model: {args.model_size}...")
    model = whisper.load_model(args.model_size, device=device)
    print("Model loaded successfully.")

    # Находим поддиректории для каждого исходного аудио
    # (например, 'Bill_Gates', 'Cameron_Russell')
    subdirectories = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    if not subdirectories:
        print(f"Error: No subdirectories found in '{args.input_dir}'. Expected format is 'input_dir/audio_name/chunks.wav'.")
        return

    # Обработка каждой поддиректории
    for subdir_name in subdirectories:
        os.makedirs(os.path.join(args.output_dir, subdir_name), exist_ok=True)
        print(f"\nProcessing audio: {subdir_name}")

        full_subdir_path = os.path.join(args.input_dir, subdir_name)

        # Находим все .wav файлы и сортируем их по номеру чанка
        chunk_files = glob.glob(os.path.join(full_subdir_path, '*.wav'))
        chunk_files.sort(key=get_chunk_number)

        if not chunk_files:
            print("  No .wav files found, skipping.")
            continue

        print(f"  Found {len(chunk_files)} chunks to transcribe.")

        # Собираем транскрипцию из всех чанков
        full_transcription = []
        for chunk_path in tqdm(chunk_files, desc=f"  Transcribing {subdir_name}"):
            # Выполняем транскрибацию
            result = model.transcribe(chunk_path, language=args.language)

            # Сохраняем текст чанка в файл
            output_file_name = os.path.basename(chunk_path).replace('.wav', '.txt')
            output_filepath = os.path.join(args.output_dir, subdir_name, output_file_name)

            # hardcode for the 1st chunk for Cameron_Russell
            if output_file_name == "Cameron_Russell-chunk-1.txt":
                result['text'] = ""

            # Сохраняем текст чанка в файл
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(result['text'].strip())

            # Добавляем полученный текст в список, удаляя лишние пробелы по краям
            full_transcription.append(result['text'].strip())

        # Объединяем все части в один текст
        final_text = " ".join(full_transcription).strip()

        # Сохраняем итоговый текст в файл (полная транскрипция всего аудио)
        output_filepath = os.path.join(args.output_dir, f"{subdir_name}.txt")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(final_text)

        print(f"  Transcription complete. Saved to: {output_filepath}")

    print("\nAll tasks finished.")


if __name__ == '__main__':
    main()

# Usage example:
# python transcribe_chunks.py -i splited_audio_robust -o transcriptions -m base -l en
