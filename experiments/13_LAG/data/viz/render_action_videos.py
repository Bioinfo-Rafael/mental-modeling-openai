"""Append synchronized A-team action plots to the three supplied recordings.

Run with the repository's .venv/bin/python. Requires matplotlib, numpy, Pillow,
ffmpeg and ffprobe. Originals are read only; outputs go to viz/action_videos/.
CSV action at t applies over [t, t+0.2); launch markers use ACMI appearance time.
The original 10 fps flight frames are repeated at 20 fps for 4x playback.
"""

from pathlib import Path
import bisect
import csv
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw" / "20261005_LAG_2v2_ACMI"
OUT = HERE / "action_videos"
UIDS = ("A0100", "A0200")
COLORS = ("#ce354c", "#7351ba")
INK = "#203044"
SPECS = [
    ("altitude_action", "ALTITUDE COMMAND", {"down": -1, "hold": 0, "up": 1}, ["Down", "Hold", "Up"]),
    ("heading_action", "HEADING COMMAND", {"left30": -2, "left15": -1, "hold": 0, "right15": 1, "right30": 2}, ["Left 30", "Left 15", "Hold", "Right 15", "Right 30"]),
    ("speed_action", "SPEED COMMAND", {"down": -1, "hold": 0, "up": 1}, ["Decel", "Hold", "Accel"]),
    ("shoot_action", "SHOOT REQUEST", {"no": 0, "yes": 1}, ["No", "Yes"]),
]
FONT = font_manager.findfont("DejaVu Sans")
BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)
    ]))["streams"][0]


def read_data(stem):
    with (RAW / "actions" / f"{stem}_actions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    groups = {uid: [r for r in rows if r["uid"] == uid] for uid in UIDS}
    launches, seen, states = {uid: [] for uid in UIDS}, set(), {}
    time = 0.0
    for line in (RAW / f"{stem}.acmi").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("#"):
            time = float(line[1:])
        elif ",T=" in line:
            uid, values = line.split(",T=", 1)
            if uid in UIDS:
                states[time, uid] = list(map(float, values.split(",")[0].split("|")))
            if "Name=AIM" in line and uid not in seen and uid[:-1] in UIDS:
                launches[uid[:-1]].append(time)
                seen.add(uid)
    columns = ("lon_deg", "lat_deg", "alt_m", "roll_deg", "pitch_deg", "yaw_deg")
    for uid, records in groups.items():
        times = np.array([float(r["time_s"]) for r in records])
        assert np.allclose(np.diff(times), 0.2), "Nonuniform action timestamps"
        for r in records:
            if r["alive"] == "1":
                values = states[float(r["time_s"]), uid]
                assert np.allclose(values, [float(r[c]) for c in columns], atol=1e-8, rtol=0)
        assert sum(bool(r["missile_launched"]) for r in records) == len(launches[uid])
    return groups, launches, time


def panel(groups, launches, end):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(9.6, 10.8), dpi=100, facecolor="#f6f8fc")
    axes = []
    # Pixel positions leave room for a title and per-frame selected-action labels.
    for i, (column, title, mapping, labels) in enumerate(SPECS):
        top, height = 185 + i * 205, 127
        ax = fig.add_axes([0.125, 1 - (top + height) / 1080, 0.84, height / 1080])
        axes.append(ax)
        for j, uid in enumerate(UIDS):
            records = groups[uid]
            times = [float(r["time_s"]) for r in records]
            ys = [mapping.get(r[column], np.nan) if r["alive"] == "1" else np.nan for r in records]
            # Include the last action's full interval, but never interpolate categories.
            times.append(times[-1] + 0.2)
            ys.append(ys[-1])
            ax.step(times, ys, where="post", color=COLORS[j], lw=2.5 if j == 0 else 1.8,
                    linestyle="-" if j == 0 else (0, (4, 2)), alpha=0.85, zorder=3+j)
            if i == 3:
                ax.scatter(launches[uid], [1.4+j*0.35]*len(launches[uid]),
                           marker="v", s=60, color=COLORS[j], zorder=6)
        ax.set_xlim(0, end)
        ticks = sorted(mapping.values())
        ax.set_yticks(ticks, labels)
        ax.set_ylim(ticks[0]-0.35, ticks[-1]+(1.0 if i == 3 else 0.35))
        ax.set_xticks(np.arange(0, end+0.01, 10))
        ax.tick_params(axis="both", labelsize=10, colors="#536277", length=0, pad=7)
        ax.grid(True, color="#dfe5ee", lw=0.7)
        for spine in ax.spines.values():
            spine.set_color("#dfe5ee")
    fig.canvas.draw()
    result = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    geometry = [(a.bbox.x0, 1080-a.bbox.y1, a.bbox.x1, 1080-a.bbox.y0) for a in axes]
    plt.close(fig)
    draw = ImageDraw.Draw(result)
    draw.text((40, 24), "A-TEAM / SELECTED ACTIONS", font=font(25, True), fill=INK)
    draw.text((40, 62), "Commands to the controller", font=font(16), fill="#657388")
    for j, uid in enumerate(UIDS):
        x = 40+j*225
        if j == 0:
            draw.line((x, 107, x+42, 107), fill=COLORS[j], width=4)
        else:
            for dx in range(0, 42, 12):
                draw.line((x+dx, 107, x+dx+7, 107), fill=COLORS[j], width=3)
        draw.text((x+52, 96), uid, font=font(18, True), fill=COLORS[j])
    for i, (_, title, _, _) in enumerate(SPECS):
        draw.text((40, 140+i*205), title, font=font(16, True), fill=INK)
    draw.text((340, 959), "Simulation time (seconds)", font=font(16), fill=INK)
    draw.text((40, 999), "Triangles: actual missile appearance in ACMI", font=font(15), fill="#657388")
    draw.text((40, 1028), "Missing actions after aircraft loss are left blank.", font=font(15), fill="#657388")
    return result, geometry


def render(stem):
    source = RAW / "recordings" / f"{stem}.mp4"
    info = probe(source)
    w, h, n = int(info["width"]), int(info["height"]), int(info["nb_frames"])
    assert info["r_frame_rate"] == "10/1" and w == 960 and h <= 940
    groups, launches, end = read_data(stem)
    chart, geometry = panel(groups, launches, end)
    base = Image.new("RGB", (1920, 1080), "white")
    base.paste(chart, (960, 0))
    draw = ImageDraw.Draw(base)
    title = "MHI v1.3 | IC 100601 | WIN" if stem.startswith("mhi") else (
        "OURS | IC 100606 | WIN" if "100606" in stem else "OURS | IC 100601 | LOSS")
    draw.text((30, 18), title, font=font(24, True), fill=INK)
    draw.text((30, 51), "Original flight recording  /  4x playback", font=font(16), fill="#657388")
    draw.line((960, 0, 960, 1080), fill="#dfe5ee", width=2)
    draw.text((30, 1023), "Flight view: 0.4 s samples  |  Action plots: 0.2 s samples", font=font(17), fill="#657388")
    ypos = 78 + (940-h)//2
    target = OUT / f"{stem}_with_actions.mp4"
    pending = target.with_suffix(".partial.mp4")
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(source),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE)
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", "20", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt",
        "yuv420p", "-movflags", "+faststart", str(pending)], stdin=subprocess.PIPE)
    times = {uid: [float(r["time_s"]) for r in records] for uid, records in groups.items()}
    try:
        for frame in range(n*2):
            if frame % 2 == 0:
                data = decoder.stdout.read(w*h*3)
                if len(data) != w*h*3:
                    raise RuntimeError("Incomplete source frame")
                flight = Image.frombytes("RGB", (w, h), data)
            t = min(round(frame*0.2, 6), end)
            canvas = base.copy()
            canvas.paste(flight, (0, ypos))
            d = ImageDraw.Draw(canvas)
            d.text((1610, 61), f"t = {t:5.1f} s", font=font(24, True), fill=INK)
            for i, (column, _, mapping, labels) in enumerate(SPECS):
                x0, y0, x1, y1 = geometry[i]
                x = 960+x0+(x1-x0)*t/end
                d.line((x, y0, x, y1), fill=INK, width=2)
                for j, uid in enumerate(UIDS):
                    idx = bisect.bisect_right(times[uid], t+1e-7)-1
                    r = groups[uid][idx]
                    if t >= times[uid][-1]+0.2-1e-7:
                        label = "END"
                    elif r["alive"] != "1":
                        label = "INACTIVE"
                    else:
                        label = r[column].upper()
                    d.text((960+355+j*280, 140+i*205), f"{uid}  {label}", font=font(15, True), fill=COLORS[j])
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        assert decoder.wait() == 0
        assert encoder.wait() == 0
        check = probe(pending)
        assert int(check["nb_frames"]) == n*2
        assert check["r_frame_rate"] == "20/1"
        assert abs(float(check["duration"])-float(info["duration"])) < 0.001
        pending.replace(target)
        print(json.dumps({"file": str(target), "frames": n*2, "duration": check["duration"],
                          "simulation_end": end, "launches": launches}), flush=True)
    finally:
        for proc in (decoder, encoder):
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.rglob("*") if p.is_file()}
    for recording in sorted((RAW / "recordings").glob("*.mp4")):
        render(recording.stem)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in originals.items())
    print("Verified: all original files unchanged.", flush=True)
