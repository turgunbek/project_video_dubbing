from TTS.api import TTS
import os
import glob
from tqdm import tqdm
import re
import html
import torch
import torchaudio
import torchaudio.transforms as T
from silero_vad import get_speech_timestamps
from pydub import AudioSegment
import tempfile


# Функция для очистки текста от лишних символов
def clean_text(text: str) -> str:
    text = html.unescape(text)
    bad_chars = "Ќ¬€"
    text = ''.join(c for c in text if c not in bad_chars)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Функция для правильной сортировки чанков
def get_chunk_number(filename: str) -> int:
    try:
        base = os.path.basename(filename)
        num_str = base.split('-')[-1].replace('.wav', '')
        return int(num_str)
    except (IndexError, ValueError):
        return 0


def get_speech_start_time_in_chunk(audio_path, vad_model):
    """Определяет время начала речи в чанке с помощью VAD"""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        speech_timestamps = get_speech_timestamps(waveform, vad_model, return_seconds=True)
        if speech_timestamps:
            return speech_timestamps[0]['start']
        return None
    except Exception as e:
        print(f"  > VAD Error on {os.path.basename(audio_path)}: {e}")
        return None


def create_synced_audio(original_chunk_path, russian_text, tts_model, voice_prompt_wav, vad_model):
    """
    Создает синхронизированный аудиочанк:
    - Определяет начало речи в оригинальном чанке
    - Генерирует TTS для русского текста
    - Объединяет: тишина/музыка из начала оригинала + TTS речь
    """
    # 1. Определяем начало речи в оригинальном чанке
    speech_start_time = get_speech_start_time_in_chunk(original_chunk_path, vad_model)

    # 2. Загружаем оригинальный чанк
    original_audio = AudioSegment.from_wav(original_chunk_path)

    # 3. Генерируем TTS для русского текста
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_tts_path = temp_file.name

    try:
        tts_model.tts_to_file(
            text=russian_text,
            file_path=temp_tts_path,
            speaker_wav=voice_prompt_wav,
            language="ru"
        )

        # 4. Загружаем сгенерированный TTS
        tts_audio = AudioSegment.from_wav(temp_tts_path)

        # 5. Создаем финальный синхронизированный аудио
        if speech_start_time is not None and speech_start_time > 0.1:
            # Есть музыка/тишина в начале
            speech_start_ms = int(speech_start_time * 1000)

            # Берем начальную часть (музыка/тишина) из оригинала
            intro_part = original_audio[:speech_start_ms]

            # Объединяем: начальная часть + TTS речь
            synced_audio = intro_part + tts_audio

            print(f"    Created hybrid audio: {speech_start_time:.2f}s intro + {len(tts_audio)/1000:.2f}s TTS")

        else:
            # Речь начинается сразу, используем только TTS
            synced_audio = tts_audio
            print(f"    Created TTS-only audio: {len(tts_audio)/1000:.2f}s")

        return synced_audio

    finally:
        # Удаляем временный файл
        if os.path.exists(temp_tts_path):
            os.unlink(temp_tts_path)


def main(translations_dir, output_dir, audios_dir):
    """Функция для обработки всех файлов в папке с использованием TTS."""

    # --- Инициализация модели TTS ---
    print("Loading TTS model (XTTSv2)...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    # --- Инициализация VAD модели ---
    print("Loading Silero VAD model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        onnx=False
    )

    subdirectories = [d for d in os.listdir(translations_dir) if os.path.isdir(os.path.join(translations_dir, d))]

    for subdir_name in subdirectories:
        print(f"\nProcessing audio: {subdir_name}")

        # Путь к чанкам и образцу голоса для одного спикера
        chunks_dir = os.path.join(audios_dir, subdir_name)
        output_dir_dubbed = os.path.join(output_dir, subdir_name)
        os.makedirs(output_dir_dubbed, exist_ok=True)

        # ОПРЕДЕЛЯЕМ НАДЕЖНЫЙ "ЗОЛОТОЙ" ПРОМПТ
        if subdir_name == "Bill_Gates":
            voice_prompt_wav = os.path.join(chunks_dir, subdir_name + "-chunk-3.wav")
        elif subdir_name == "Cameron_Russell":
            voice_prompt_wav = os.path.join(chunks_dir, subdir_name + "-chunk-5.wav")
        else:
            # Для других спикеров используем первый чанк
            # voice_prompt_wav = os.path.join(chunks_dir, subdir_name + "-chunk-1.wav")
            print("Find the best chunk for voice prompt manually.")
            pass  # Пропускаем, если не определен

        # 1. ЗАГРУЖАЕМ ВСЕ ПЕРЕВОДЫ В ПАМЯТЬ (в словарь)
        print("Loading translations...")
        translations = {}
        translation_dir = os.path.join(translations_dir, subdir_name)
        translation_files = glob.glob(os.path.join(translation_dir, "*.txt"))

        for t_file in translation_files:
            with open(t_file, 'r', encoding='utf-8') as f:
                key = os.path.splitext(os.path.basename(t_file))[0]
                translations[key] = f.read().strip()

        print(f"Loaded {len(translations)} translation files.")

        # --- Основной цикл ---
        print("Starting dubbing process...")
        chunk_files = glob.glob(os.path.join(chunks_dir, '*.wav'))
        chunk_files.sort(key=get_chunk_number)

        for chunk_path in tqdm(chunk_files, desc="Processing pipeline"):
            chunk_key = os.path.splitext(os.path.basename(chunk_path))[0]
            russian_text = translations.get(chunk_key)

            if not russian_text:
                print(f"  > Warning: No translation found for {chunk_key}. Skipping.")
                continue

            # --- ОЧИСТКА ТЕКСТА ---
            clean_russian_text = clean_text(russian_text)

            # Добавляем отступ в начале только если текст не пустой
            if clean_russian_text:
                if clean_russian_text.endswith(('.', '!', '?')):
                    padded_text = f"... {clean_russian_text}"
                else:
                    padded_text = f"... {clean_russian_text}."
            else:
                print(f"  > Warning: Empty translation for {chunk_key}. Skipping.")
                continue

            # --- СОЗДАНИЕ СИНХРОНИЗИРОВАННОГО АУДИО ---
            print(f"  Processing {chunk_key}...")

            try:
                synced_audio = create_synced_audio(
                    chunk_path,
                    padded_text,
                    tts_model,
                    voice_prompt_wav,
                    vad_model
                )

                # Сохраняем результат
                output_wav_path = os.path.join(output_dir_dubbed, os.path.basename(chunk_path))
                synced_audio.export(output_wav_path, format="wav")

            except Exception as e:
                print(f"  > Error processing {chunk_key}: {e}")
                continue

        print(f"Dubbing process complete for speaker {subdir_name}!")


if __name__ == '__main__':
    main(translations_dir="translations",
         output_dir="dubbed_audio",
         audios_dir="splited_audio_robust")

# Usage example:
# python tts_generate_dub.py
