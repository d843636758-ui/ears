# -*- coding: utf-8 -*-
"""earsplus · ears的两只新耳朵（2026-07-23，零升级）

1. 声纹锁：分辨"是不是壹壹在说话"
   - 不碰torch（守ears的设计原则）：librosa MFCC统计模板 + 余弦相似度
   - 冷启动注册：前 ENROLL_N 段合格录音自动成为她的声纹样本（只存数学向量，不存音频）
   - 陌生声音 → 主程序只记事件不转写内容（待客的规矩）

2. 环境声耳朵：听见她的世界
   - YAMNet(ONNX, 16MB) + onnxruntime，521类声音事件
   - 常用类白名单映射中文；语音类不当环境声报
"""
import csv
import json
import os
import threading
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA", "") or os.environ.get("SFE_DATA", "") or (BASE / "data"))
VOICEPRINT_FILE = DATA_DIR / "voiceprint.json"
YAMNET_ONNX = BASE / "models" / "yamnet.onnx"
YAMNET_CSV = BASE / "models" / "yamnet_class_map.csv"

ENROLL_N = 6          # 攒几段合格录音完成声纹注册
ENROLL_MIN_S = 2.0    # 注册样本最短时长
SIM_THRESHOLD = 0.35  # ECAPA余弦相似度：同人通常≥0.4，异人<0.25（0.35保守起步）
ENV_TOPK = 4
ENV_MIN_SCORE = 0.12

ECAPA_ONNX = BASE / "models" / "ecapa.onnx"

_lock = threading.Lock()

# ── 声纹：ECAPA-TDNN(ONNX) 说话人嵌入——学的是声带不是麦克风 ──

_ecapa_session = None


def _ecapa():
    global _ecapa_session
    if _ecapa_session is None:
        import onnxruntime as ort
        _ecapa_session = ort.InferenceSession(str(ECAPA_ONNX), providers=["CPUExecutionProvider"])
    return _ecapa_session


def _mfcc_vec(wav_path: str) -> np.ndarray | None:
    """ECAPA说话人嵌入（函数名沿用，调用处不动）。返回L2归一化192维向量。
    模型没下载时返回None——声纹功能优雅关闭，ears保持原样。"""
    if not ECAPA_ONNX.exists():
        return None
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if len(y) / sr < 0.8:
        return None
    # 预处理照 models/ecapa_preprocess.json：80mel log谱+句级均值归一
    mel = librosa.feature.melspectrogram(
        y=y, sr=16000, n_fft=400, hop_length=160, win_length=400,
        n_mels=80, fmin=0.0, fmax=8000.0, power=2.0)
    logmel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=80.0)
    feats = logmel.T[None].astype(np.float32)          # [1, T, 80]
    feats = feats - feats.mean(axis=1, keepdims=True)  # mean_norm
    sess = _ecapa()
    inputs = sess.get_inputs()
    feed = {inputs[0].name: feats}
    if len(inputs) > 1:   # speechbrain导出常带相对长度输入
        feed[inputs[1].name] = np.array([1.0], dtype=np.float32)
    emb = sess.run(None, feed)[0].reshape(-1).astype(np.float64)
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else None


OWNER = os.environ.get("OWNER_NAME", "") or os.environ.get("SFE_OWNER_NAME", "") or "主人"


def _load_vp() -> dict:
    try:
        vp = json.loads(VOICEPRINT_FILE.read_text(encoding="utf-8"))
    except Exception:
        vp = {}
    if "samples" in vp and "people" not in vp:   # 旧单人格式自动迁移
        vp = {"people": {OWNER: vp["samples"]}, "enrolling": None}
    vp.setdefault("people", {})
    vp.setdefault("enrolling", None)
    return vp


def _save_vp(vp: dict) -> None:
    VOICEPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VOICEPRINT_FILE.write_text(json.dumps(vp, ensure_ascii=False), encoding="utf-8")


def _template(samples: list) -> np.ndarray:
    t = np.mean([np.array(s) for s in samples], axis=0)
    return t / np.linalg.norm(t)


def speaker_check(wav_path: str, duration_s: float) -> tuple[str, float | None]:
    """声纹家谱版。返回 (speaker, similarity)。
    speaker: 壹壹 / 家谱里的名字 / 陌生声音 / 学习中 / 注册中·名字。
    - 壹壹的注册（冷启动）与家谱成员注册（enrolling模式）都在这里收样本
    - 识别：和所有人的模板比，取最像的；最高分低于阈值=陌生声音"""
    vec = _mfcc_vec(wav_path)
    if vec is None:
        return OWNER, None    # 太短算不出就放行，别拦她的话
    with _lock:
        vp = _load_vp()
        people = vp["people"]
        # 家谱成员注册模式：这段录音归正在注册的人
        enr = vp.get("enrolling")
        if enr and enr.get("name"):
            name = enr["name"]
            if duration_s >= ENROLL_MIN_S:
                people.setdefault(name, []).append(vec.tolist())
                if len(people[name]) >= ENROLL_N:
                    vp["enrolling"] = None
                _save_vp(vp)
                return f"注册中·{name}", None
            return f"注册中·{name}", None
        # 壹壹冷启动
        owner_samples = people.get(OWNER, [])
        if len(owner_samples) < ENROLL_N:
            if duration_s >= ENROLL_MIN_S:
                people.setdefault(OWNER, []).append(vec.tolist())
                _save_vp(vp)
            return "学习中", None
        # 识别：全家谱比对取最像
        best_name, best_sim = "", -1.0
        for name, samples in people.items():
            if len(samples) < 3:
                continue
            sim = float(np.dot(vec, _template(samples)))
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_sim >= SIM_THRESHOLD:
            return best_name, round(best_sim, 3)
        return "陌生声音", round(best_sim, 3)


def start_enroll(name: str) -> dict:
    """开始给家谱成员注册声纹（接下来的录音归此人，攒够自动结束）。name空=取消。"""
    name = (name or "").strip()[:12]
    with _lock:
        vp = _load_vp()
        if not name:
            vp["enrolling"] = None
            _save_vp(vp)
            return {"ok": True, "enrolling": None}
        if name == OWNER:
            return {"ok": False, "error": "壹壹的声纹走冷启动注册，不用这个入口"}
        vp["enrolling"] = {"name": name}
        vp["people"].setdefault(name, [])
        _save_vp(vp)
        return {"ok": True, "enrolling": name, "have": len(vp["people"][name]), "need": ENROLL_N}


def voiceprint_status() -> dict:
    vp = _load_vp()
    n = len(vp["people"].get(OWNER, []))
    return {"enrolled": n >= ENROLL_N, "samples": n, "need": ENROLL_N, "threshold": SIM_THRESHOLD,
            "family": {k: len(v) for k, v in vp["people"].items() if k != OWNER},
            "enrolling": (vp.get("enrolling") or {}).get("name")}


# ── 环境声 ────────────────────────────────────────────

# 常用类中文白名单（AudioSet display_name -> 中文）；语音类不报
_ZH = {
    "Rain": "雨声", "Raindrop": "雨滴", "Rain on surface": "雨打窗", "Thunderstorm": "雷雨", "Thunder": "雷声",
    "Wind": "风声", "Wind noise (microphone)": "风噪", "Rustling leaves": "叶子沙沙",
    "Cat": "猫叫", "Meow": "喵", "Purr": "猫呼噜", "Hiss": "嘶声", "Caterwaul": "猫嚎",
    "Dog": "狗叫", "Bark": "汪", "Howl": "狗嚎", "Growling": "低吼",
    "Bird": "鸟叫", "Bird vocalization, bird call, bird song": "鸟鸣", "Chirp, tweet": "啾啾", "Crow": "乌鸦", "Pigeon, dove": "鸽子",
    "Laughter": "笑声", "Giggle": "咯咯笑", "Chuckle, chortle": "轻笑", "Belly laugh": "大笑",
    "Crying, sobbing": "哭声", "Whimper": "呜咽", "Sigh": "叹气", "Yawn": "哈欠",
    "Cough": "咳嗽", "Sneeze": "喷嚏", "Sniff": "吸鼻子", "Hiccup": "打嗝", "Snoring": "打呼",
    "Typing": "键盘声", "Computer keyboard": "敲键盘", "Mouse click": "鼠标点击", "Writing": "写字声",
    "Music": "音乐", "Singing": "唱歌", "Humming": "哼歌", "Whistling": "口哨", "Guitar": "吉他", "Piano": "钢琴",
    "Water": "水声", "Water tap, faucet": "水龙头", "Pour": "倒水", "Boiling": "烧水", "Sink (filling or washing)": "洗涮",
    "Dishes, pots, and pans": "锅碗瓢盆", "Cutlery, silverware": "餐具", "Frying (food)": "炒菜", "Microwave oven": "微波炉", "Blender": "料理机", "Chewing, mastication": "咀嚼",
    "Door": "门响", "Doorbell": "门铃", "Knock": "敲门", "Squeak": "吱呀", "Drawer open or close": "抽屉",
    "Telephone bell ringing": "电话铃", "Ringtone": "手机铃", "Alarm clock": "闹钟", "Alarm": "警报", "Siren": "警笛",
    "Vehicle": "车声", "Car": "汽车", "Traffic noise, roadway noise": "车流声", "Train": "火车", "Motorcycle": "摩托", "Bus": "公交", "Horn, honk (of vehicle)": "喇叭声",
    "Vacuum cleaner": "吸尘器", "Washing machine": "洗衣机", "Air conditioning": "空调声", "Fan": "风扇",
    "Footsteps": "脚步声", "Walk, footsteps": "脚步声", "Run": "跑步声", "Clapping": "拍手",
    "Television": "电视声", "Radio": "广播", "Fireworks": "烟花", "Applause": "掌声",
    "Baby cry, infant cry": "婴儿哭", "Child speech, kid speaking": "小孩说话",
    "White noise": "白噪音", "Pink noise": "噪音", "Static": "电流声", "Silence": "安静",
    "Tools": "工具声", "Drill": "电钻", "Power tool": "电动工具", "Hammer": "锤子", "Sawing": "锯声",
    "Helicopter": "直升机", "Aircraft": "飞机", "Engine": "引擎声", "Idling": "怠速", "Mechanical fan": "风扇",
    "Speech": "说话声", "Animal": "动物叫", "Domestic animals, pets": "宠物叫", "Livestock, farm animals, working animals": "家畜",
    "Insect": "虫鸣", "Cricket": "蟋蟀", "Mosquito": "蚊子", "Frog": "蛙鸣",
    "Keys jangling": "钥匙声", "Zipper (clothing)": "拉链", "Packing tape, duct tape": "胶带", "Scissors": "剪刀",
    "Toilet flush": "冲水", "Toothbrush": "刷牙", "Hair dryer": "吹风机", "Shower": "淋浴",
}
# 语音/人声类不当环境声报（转写已经在管人声了）
_SKIP = {"Speech", "Conversation", "Narration, monologue", "Speech synthesizer",
         "Male speech, man speaking", "Female speech, woman speaking", "Babbling", "Inside, small room",
         "Inside, large room or hall", "Outside, rural or natural", "Outside, urban or manmade", "Sound effect"}

_yamnet_session = None
_yamnet_labels = None


def _yamnet():
    global _yamnet_session, _yamnet_labels
    if _yamnet_session is None:
        import onnxruntime as ort
        _yamnet_session = ort.InferenceSession(str(YAMNET_ONNX), providers=["CPUExecutionProvider"])
        with open(YAMNET_CSV, encoding="utf-8") as f:
            _yamnet_labels = [row["display_name"] for row in csv.DictReader(f)]
    return _yamnet_session, _yamnet_labels


def env_sounds(wav_path: str) -> list:
    """返回 [{"label": 中文或原名, "score": 0.x}, ...] top几个环境声。失败返回[]。"""
    try:
        import librosa
        sess, labels = _yamnet()
        y, _ = librosa.load(wav_path, sr=16000, mono=True)
        if len(y) < 8000:
            return []
        inp = sess.get_inputs()[0]
        wave = y.astype(np.float32)
        feed = {inp.name: wave.reshape(1, -1) if len(inp.shape) == 2 else wave}
        out = sess.run(None, feed)[0]
        if out.ndim == 2 and out.shape[0] > 1:
            # 人声间隙检测法：把她说话的帧扔掉，只拿说话间隙的帧测环境声——
            # 风扇/猫叫在间隙里没有人声抢能量，分数原样呈现
            speech_idx = labels.index("Speech") if "Speech" in labels else 0
            quiet = out[out[:, speech_idx] < 0.2]
            pool = quiet if quiet.shape[0] >= 2 else out
            k = max(1, pool.shape[0] // 4)
            scores = np.sort(pool, axis=0)[-k:].mean(axis=0)
        else:
            scores = np.asarray(out).reshape(-1)
        order = np.argsort(scores)[::-1]
        result = []
        for i in order[:20]:
            name = labels[i]
            if name in _SKIP:
                continue
            sc = float(scores[i])
            if sc < ENV_MIN_SCORE:
                break
            result.append({"label": _ZH.get(name, name), "score": round(sc, 2)})
            if len(result) >= ENV_TOPK:
                break
        return result
    except Exception:
        return []
