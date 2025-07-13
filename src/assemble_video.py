from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import json
import os


def main():
    print("Starting video assembly...")

    base_audio_dir = "splited_audio_robust"
    speakers = [d for d in os.listdir(base_audio_dir) if os.path.isdir(os.path.join(base_audio_dir, d))]

    if not speakers:
        print(f"No speaker directories found in {base_audio_dir}. Exiting.")
        return

    for subdir_name in speakers:  # ["Bill_Gates", "Cameron_Russell"]:
        print(f"\nProcessing speaker: {subdir_name}")

        # --- Входные файлы ---
        original_video_path = f"{subdir_name}.mp4"
        original_chunks_dir = f"splited_audio_robust/{subdir_name}"
        dubbed_chunks_dir = f"dubbed_audio/{subdir_name}"
        timings_file = f"splited_audio_robust/{subdir_name}/{subdir_name}_timings.json"

        # --- Выходной файл ---
        final_video_path = f"final_video/{subdir_name}_dubbed.mp4"
        os.makedirs("final_video", exist_ok=True)

        # Список для хранения ВСЕХ открытых клипов
        all_opened_clips = []

        try:
            print(f"Loading timings from {timings_file}...")
            with open(timings_file, 'r', encoding='utf-8') as f:
                timings = json.load(f)

            print(f"Loading original video: {original_video_path}...")
            video_clip = VideoFileClip(original_video_path)
            all_opened_clips.append(video_clip)

            audio_clips_for_timeline = []
            print("Preparing final audio track...")

            for timing_info in timings:
                chunk_name = timing_info["chunk_name"]
                chunk_start_time_global = timing_info["start_time"]
                original_chunk_path = os.path.join(original_chunks_dir, chunk_name)
                dubbed_chunk_path = os.path.join(dubbed_chunks_dir, chunk_name)

                # Простая логика: если есть дубляж - используем его, иначе оригинал
                if os.path.exists(dubbed_chunk_path):
                    print(f"  - Using DUBBED audio for {chunk_name}")
                    audio_clip = AudioFileClip(dubbed_chunk_path)
                elif os.path.exists(original_chunk_path):
                    print(f"  - Using ORIGINAL audio for {chunk_name}")
                    audio_clip = AudioFileClip(original_chunk_path)
                else:
                    print(f"  - Warning: No audio found for {chunk_name}, skipping")
                    continue

                # Размещаем клип на временной шкале
                audio_clip = audio_clip.with_start(chunk_start_time_global)
                audio_clips_for_timeline.append(audio_clip)
                all_opened_clips.append(audio_clip)

            # Создаем финальную аудиодорожку
            final_audio = CompositeAudioClip(audio_clips_for_timeline)
            all_opened_clips.append(final_audio)

            print("Replacing audio track...")
            final_video = video_clip.with_audio(final_audio)
            all_opened_clips.append(final_video)

            print(f"Writing final video to {final_video_path}...")
            final_video.write_videofile(
                final_video_path,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                logger='bar'
            )

            print(f"Successfully created: {final_video_path}")

        except Exception as e:
            print(f"Error processing {subdir_name}: {e}")

        finally:
            print("Closing all media files...")
            for clip in all_opened_clips:
                try:
                    clip.close()
                except Exception:
                    pass


if __name__ == '__main__':
    main()
