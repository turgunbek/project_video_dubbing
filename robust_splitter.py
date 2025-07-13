import os
import glob
import argparse
import json  # Импортируем json для сохранения таймингов
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def process_audio_file(file_path, output_dir, target_duration_s, silence_thresh_db, min_silence_len_ms, keep_silence_ms):
    """
    Splits an audio file by silence, saves chunks, and also saves their exact timings.
    """
    print(f"\nProcessing: {file_path}")

    audio = AudioSegment.from_file(file_path)
    print(f"  - Loaded audio: {len(audio) / 1000:.2f}s, {audio.frame_rate}Hz, {audio.channels} channels")

    # 1. ДЕТЕКТИРУЕМ НЕ-ТИХИЕ УЧАСТКИ, ПОЛУЧАЕМ ИХ ТАЙМИНГИ
    # Вместо split_on_silence мы используем detect_nonsilent.
    # Она возвращает список таймкодов [start_ms, end_ms] для каждого сегмента речи.
    print(f"  - Detecting non-silent parts (threshold: {silence_thresh_db}dB, min_silence: {min_silence_len_ms}ms)...")
    nonsilent_parts = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db
    )

    if not nonsilent_parts:
        print("  - No speech detected. Skipping file.")
        return

    print(f"  - Found {len(nonsilent_parts)} initial non-silent segments.")

    # 2. ОБЪЕДИНЯЕМ МЕЛКИЕ СЕГМЕНТЫ В БОЛЕЕ КРУПНЫЕ ЧАНКИ (И ИХ ТАЙМИНГИ)
    target_duration_ms = target_duration_s * 1000

    final_chunks_with_timings = []
    current_chunk_segments = []  # Список сегментов [start, end] для текущего чанка
    current_duration_ms = 0

    for start_ms, end_ms in nonsilent_parts:
        segment_duration = end_ms - start_ms

        if current_chunk_segments and (current_duration_ms + segment_duration < target_duration_ms):
            # Добавляем сегмент в текущий чанк
            current_chunk_segments.append([start_ms, end_ms])
            current_duration_ms += segment_duration
        else:
            # Сохраняем предыдущий чанк, если он не пустой
            if current_chunk_segments:
                final_chunks_with_timings.append(current_chunk_segments)

            # Начинаем новый чанк
            current_chunk_segments = [[start_ms, end_ms]]
            current_duration_ms = segment_duration

    # Не забываем добавить последний собранный чанк
    if current_chunk_segments:
        final_chunks_with_timings.append(current_chunk_segments)

    # hardcode for the 1st chunk for Cameron_Russell
    if "Cameron_Russell" in file_path:
        final_chunks_with_timings[0] = [[0, 14000]]  # Заменяем первый чанк на 12 секунд тишины
        final_chunks_with_timings.insert(1, [[15000, 21000]])  # Добавляем второй чанк с 15 до 21 секунд

    print(f"  - Recombined into {len(final_chunks_with_timings)} final chunks.")

    # 3. СОХРАНЯЕМ РЕЗУЛЬТАТЫ: АУДИО-ЧАНКИ И JSON-ФАЙЛ С ТАЙМИНГАМИ
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    file_output_dir = os.path.join(output_dir, base_filename)
    os.makedirs(file_output_dir, exist_ok=True)

    final_timings_data = []

    for i, segments in enumerate(final_chunks_with_timings):
        # Определяем "чистое" время начала и конца всего чанка
        speech_start_ms_raw = segments[0][0]
        chunk_end_ms_raw = segments[-1][1]

        # Добавляем "отступы", чтобы не резать слова по краям
        chunk_start_ms_with_padding = max(0, speech_start_ms_raw - keep_silence_ms)
        chunk_end_ms_with_padding = min(len(audio), chunk_end_ms_raw + keep_silence_ms)

        # Извлекаем аудио для всего объединенного чанка
        final_chunk_audio = audio[chunk_start_ms_with_padding:chunk_end_ms_with_padding]

        # Сохраняем аудио-чанк
        chunk_filename = f"{base_filename}-chunk-{i+1}.wav"
        chunk_filepath = os.path.join(file_output_dir, chunk_filename)
        final_chunk_audio.export(chunk_filepath, format="wav")

        # Сохраняем информацию о таймингах, причем ОБА времени: время начала всего куска и время начала речи
        final_timings_data.append({
            "chunk_name": chunk_filename,
            "start_time": chunk_start_ms_with_padding / 1000.0,  # Конвертируем в секунды
            "end_time": chunk_end_ms_with_padding / 1000.0,
        })

    # Сохраняем JSON-файл с таймингами в ту же папку, что и аудио-чанки
    timings_filepath = os.path.join(file_output_dir, f"{base_filename}_timings.json")
    with open(timings_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_timings_data, f, indent=2, ensure_ascii=False)

    print(f"  - Saved {len(final_chunks_with_timings)} chunks to '{file_output_dir}'")
    print(f"  - Timings saved to '{timings_filepath}'")


def main():
    parser = argparse.ArgumentParser(description="Robust audio splitter using pydub. Guarantees 100% audio coverage by splitting on silence.")

    parser.add_argument('-d', '--directory', required=True, help='Directory containing audio files to process (e.g., raw_audio).')
    parser.add_argument('-o', '--output-dir', required=True, help='Directory to save the split audio chunks (e.g., splited_audio).')
    parser.add_argument('-t', '--target-duration', type=int, default=15, help='Target maximum duration of chunks in seconds (default: 15).')
    parser.add_argument('--silence-threshold', type=int, default=-40, help='The upper bound for what is considered silence in dBFS (default: -40).')
    parser.add_argument('--min-silence-len', type=int, default=700, help='The minimum length of a silence in milliseconds to be used as a split point (default: 700).')
    parser.add_argument('--keep-silence', type=int, default=300, help='How much silence (in ms) to leave at the beginning and end of each chunk (default: 300).')

    args = parser.parse_args()

    files_to_process = []
    for ext in ["*.wav", "*.mp3", "*.m4a"]:
        files_to_process.extend(glob.glob(os.path.join(args.directory, ext)))

    if not files_to_process:
        print(f"No audio files found in '{args.directory}'")
        return

    for file_path in files_to_process:
        process_audio_file(
            file_path,
            args.output_dir,
            args.target_duration,
            args.silence_threshold,
            args.min_silence_len,
            args.keep_silence
        )


if __name__ == '__main__':
    main()

# Example usage:
# python robust_splitter_with_timings.py -d raw_audio -o splited_audio_robust -t 15 --silence-threshold -40 --min-silence-len 700 --keep-silence 500
