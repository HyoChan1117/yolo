import os
import subprocess
import json

video_dir = "video"
save_dir = "human/datasets"

os.makedirs(save_dir, exist_ok=True)

video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

total_saved = 0

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    prefix = os.path.splitext(video_file)[0]

    # fps 읽기
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True
    )
    fps = 30.0
    try:
        info = json.loads(probe.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                r = stream.get("r_frame_rate", "30/1").split("/")
                fps = float(r[0]) / float(r[1])
                break
    except Exception:
        pass

    interval = max(1, int(fps * 10))  # 10초마다

    # ffmpeg로 전체 프레임 추출 후 interval 적용
    tmp_dir = os.path.join(save_dir, f"_tmp_{prefix}")
    os.makedirs(tmp_dir, exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, os.path.join(tmp_dir, "frame_%06d.jpg")],
        capture_output=True
    )

    frames = sorted(os.listdir(tmp_dir))
    save_count = 0
    for i, fname in enumerate(frames):
        if i % interval == 0:
            src = os.path.join(tmp_dir, fname)
            dst = os.path.join(save_dir, f"{prefix}_frame_{save_count:04d}.jpg")
            os.rename(src, dst)
            save_count += 1
        else:
            os.remove(os.path.join(tmp_dir, fname))

    os.rmdir(tmp_dir)
    print(f"{video_file}: {save_count}장 저장 (총 {len(frames)}프레임, {fps:.1f}fps)")
    total_saved += save_count

print(f"\n전체 저장 완료: {total_saved}장")