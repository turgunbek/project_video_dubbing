# app.py
import streamlit as st
import os
import glob
from pathlib import Path
# import torch

# --- Конфигурация страницы и заголовки ---
st.set_page_config(layout="wide", page_title="Демонстрация пайплайна видео-дубляжа")
st.title("🎬 Демонстрация пайплайна видео-дубляжа")
st.write("Этот проект демонстрирует полный цикл дублирования видео: от извлечения аудио до генерации синхронизированной русской озвучки с помощью Zero-shot TTS.")

# --- Глобальные переменные и пути ---
ARTIFACTS_ROOT = "pipeline_artifacts"
LIVE_DEMO_SAMPLES_ROOT = "live_demo_samples"
FINAL_VIDEO_ROOT = "final_video"
SPEAKERS = ["Bill_Gates", "Cameron_Russell"]


# --- Кэширование для тяжелых моделей ---
# Это гарантирует, что модели загружаются только один раз
@st.cache_resource
def load_whisper_model():
    import whisper
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Whisper model loaded.")
    return model


@st.cache_resource
def load_translation_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("Loading Translation model...")
    model_name = "Helsinki-NLP/opus-mt-en-ru"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Translation model loaded.")
    return tokenizer, model


@st.cache_resource
def load_tts_model():
    from TTS.api import TTS
    print("Loading TTS model...")
    # Указываем gpu=False, так как на Streamlit Cloud обычно нет GPU
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("TTS model loaded.")
    return model


# --- Функции для Live Демо ---
def run_live_pipeline(audio_path, voice_prompt_path):
    # 1. Транскрибация
    st.write("Шаг 1: Транскрибация аудио...")
    placeholder1 = st.empty()
    placeholder1.info("Загрузка модели Whisper...")
    model_whisper = load_whisper_model()
    placeholder1.info("Транскрибация...")
    result = model_whisper.transcribe(audio_path)
    english_text = result['text']
    placeholder1.success("Транскрибация завершена!")
    st.text_area("Полученный английский текст:", english_text, height=100)

    # 2. Перевод
    st.write("Шаг 2: Перевод на русский язык...")
    placeholder2 = st.empty()
    placeholder2.info("Загрузка модели перевода...")
    tokenizer_mt, model_mt = load_translation_model()
    placeholder2.info("Перевод...")
    input_ids = tokenizer_mt.encode(english_text, return_tensors="pt")
    outputs = model_mt.generate(input_ids)
    russian_text = tokenizer_mt.decode(outputs[0], skip_special_tokens=True)
    placeholder2.success("Перевод завершен!")
    st.text_area("Полученный русский текст:", russian_text, height=100)

    # 3. Генерация речи (TTS)
    st.write("Шаг 3: Генерация русской озвучки (TTS)...")
    placeholder3 = st.empty()
    placeholder3.info("Загрузка модели TTS (это может занять несколько минут)...")
    model_tts = load_tts_model()
    placeholder3.warning("Генерация аудио... Пожалуйста, подождите.")

    # Добавляем интонационную паузу
    padded_text = f"... {russian_text}"

    output_tts_path = "live_demo_output.wav"
    model_tts.tts_to_file(
        text=padded_text,
        file_path=output_tts_path,
        speaker_wav=voice_prompt_path,
        language="ru"
    )
    placeholder3.success("Генерация аудио завершена!")
    st.audio(output_tts_path)

    return output_tts_path


# --- Основной интерфейс ---

tab1, tab2 = st.tabs(["🔍 Исследование готовых результатов", "🚀 Live Демо"])

with tab1:
    st.header("Исследование готовых результатов пайплайна")

    selected_speaker = st.selectbox("Выберите спикера для исследования:", SPEAKERS, key="speaker_select")

    if selected_speaker:
        st.subheader(f"Финальный результат для: {selected_speaker}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Оригинальное видео")
            original_video_path = f"{selected_speaker}.mp4"
            if os.path.exists(original_video_path):
                st.video(original_video_path)
            else:
                st.warning("Файл оригинального видео не найден.")

        with col2:
            st.markdown("##### Видео с русской озвучкой")
            dubbed_video_path = os.path.join(FINAL_VIDEO_ROOT, f"{selected_speaker}_dubbed.mp4")
            if os.path.exists(dubbed_video_path):
                st.video(dubbed_video_path)
            else:
                st.warning("Файл дублированного видео не найден.")

        st.divider()
        st.subheader("Пошаговое исследование артефактов")

        # --- Динамический выбор чанка ---
        speaker_artifacts_path = os.path.join(ARTIFACTS_ROOT, selected_speaker)
        original_chunks_path = os.path.join(speaker_artifacts_path, "splited_audio_robust")

        if os.path.exists(original_chunks_path):
            # Находим все чанки и сортируем их
            chunk_files = glob.glob(os.path.join(original_chunks_path, "*.wav"))
            chunk_files.sort(key=lambda x: int(Path(x).stem.split('-')[-1]))

            if chunk_files:
                chunk_numbers = [Path(f).stem for f in chunk_files]
                selected_chunk_name = st.select_slider(
                    "Выберите чанк для анализа:",
                    options=chunk_numbers
                )

                # --- Отображение информации о чанке ---
                chunk_num_str = selected_chunk_name.split('-')[-1]

                # 1. Оригинальный аудиочанк
                st.markdown("##### 1. Оригинальный аудиочанк (после сегментации)")
                orig_audio_path = os.path.join(original_chunks_path, f"{selected_chunk_name}.wav")
                if os.path.exists(orig_audio_path):
                    st.audio(orig_audio_path)
                else:
                    st.error("Файл не найден.")

                # 2. Транскрибация
                st.markdown("##### 2. Транскрибация (Whisper)")
                transcription_path = os.path.join(speaker_artifacts_path, "transcriptions", f"{selected_chunk_name}.txt")
                if os.path.exists(transcription_path):
                    with open(transcription_path, 'r', encoding='utf-8') as f:
                        st.text_area("Текст на английском:", f.read(), height=100, key=f"trans_{selected_chunk_name}")
                else:
                    st.info("Транскрибация для этого чанка отсутствует (вероятно, в нем нет речи).")

                # 3. Перевод
                st.markdown("##### 3. Перевод (Helsinki-NLP)")
                translation_path = os.path.join(speaker_artifacts_path, "translations", f"{selected_chunk_name}.txt")
                if os.path.exists(translation_path):
                    with open(translation_path, 'r', encoding='utf-8') as f:
                        st.text_area("Текст на русском:", f.read(), height=100, key=f"trans_ru_{selected_chunk_name}")
                else:
                    st.info("Перевод для этого чанка отсутствует.")

                # 4. Дубляж
                st.markdown("##### 4. Сгенерированный дубляж (XTTSv2)")
                dubbed_audio_path = os.path.join(speaker_artifacts_path, "dubbed_audio", f"{selected_chunk_name}.wav")
                if os.path.exists(dubbed_audio_path):
                    st.audio(dubbed_audio_path)
                else:
                    st.info("Дублированный аудиофайл для этого чанка отсутствует.")
            else:
                st.warning("Не найдено аудиочанков для анализа.")
        else:
            st.error(f"Не найдена папка с артефактами: {original_chunks_path}")


with tab2:
    st.header("Live Демо на коротком аудиофрагменте")
    st.info("Здесь вы можете запустить пайплайн в реальном времени на небольшом аудиофайле. Обработка может занять несколько минут.")

    sample_files = glob.glob(os.path.join(LIVE_DEMO_SAMPLES_ROOT, "*.wav"))
    sample_names = [Path(f).name for f in sample_files]

    if sample_files:
        selected_sample_name = st.selectbox("Выберите аудиофрагмент для обработки:", sample_names)
        selected_sample_path = os.path.join(LIVE_DEMO_SAMPLES_ROOT, selected_sample_name)

        st.write("Исходный аудиофрагмент:")
        st.audio(selected_sample_path)

        st.write("Для клонирования голоса необходим образец. Выберите спикера, чей голос использовать:")
        voice_speaker = st.selectbox("Выберите голос:", SPEAKERS, key="voice_select")

        # Определяем путь к "золотому" промпту
        if voice_speaker == "Bill_Gates":
            voice_prompt_path = os.path.join(ARTIFACTS_ROOT, voice_speaker, "splited_audio_robust", f"{voice_speaker}-chunk-3.wav")
        elif voice_speaker == "Cameron_Russell":
            voice_prompt_path = os.path.join(ARTIFACTS_ROOT, voice_speaker, "splited_audio_robust", f"{voice_speaker}-chunk-5.wav")

        st.write(f"Будет использован голос из файла: `{os.path.basename(voice_prompt_path)}`")
        st.audio(voice_prompt_path)

        if st.button("🚀 Запустить пайплайн!"):
            with st.spinner("Выполняется обработка... Это может занять до 5 минут."):
                run_live_pipeline(selected_sample_path, voice_prompt_path)

    else:
        st.warning(f"Не найдены аудиосемплы в папке '{LIVE_DEMO_SAMPLES_ROOT}'.")
