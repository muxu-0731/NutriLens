# ==============================================
# NutriLens 智能饮食识别与慢病管理系统
# 核心后端程序 (Flask + YOLOv8 + 营养数据库 + SQLite)
# 功能：实现食物AI识别、分病种营养分析、饮食记录持久化、周报统计
# ==============================================

import base64
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from json.decoder import JSONDecoder
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import requests
except ImportError:
    requests = None

# 禁用 CUDA GPU 加速，强制使用 CPU 运行（适配 Jetson/普通电脑）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, jsonify, render_template, request

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# 2. 初始化 Flask Web 应用实例
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "nutrilens.db"


def load_env_file(env_path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def get_env_value(key, default=""):
    return str(os.environ.get(key, default)).strip()


load_env_file(BASE_DIR / ".env")

ARK_API_KEY = get_env_value("ARK_API_KEY")
ARK_VISION_URL = get_env_value("ARK_VISION_URL", "https://ark.cn-beijing.volces.com/api/v3/responses")
ARK_VISION_MODEL_ENDPOINT = get_env_value("ARK_VISION_MODEL_ENDPOINT")
ARK_CHAT_URL = get_env_value("ARK_CHAT_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
ARK_TEXT_MODEL_ENDPOINT = get_env_value("ARK_TEXT_MODEL_ENDPOINT")

ARK_REQUEST_TIMEOUT = 20
YOLO_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_FULL_PORTION_GRAMS = 200
DEFAULT_VISUAL_TIP = "常规一盘"
RECOMMENDED_RATIO_TARGET = 0.30
DEFAULT_USER_ID = "anonymous"

MEAL_FACTORS = {
    "breakfast": 0.30,
    "lunch": 0.40,
    "dinner": 0.30,
    "snack": 0.10
}

DISEASE_CN_MAP = {
    "diabetes": "糖尿病",
    "hyperglycemia": "高血糖",
    "hyperlipidemia": "高血脂",
    "hypertension": "高血压"
}

SCALABLE_NUTRITION_FIELDS = [
    "calorie",
    "total_carbs",
    "protein",
    "fat",
    "net_carbs",
    "dietary_fiber",
    "gl",
    "sodium",
    "added_sugar",
    "saturated_fat",
    "cholesterol",
    "salt_equivalent"
]

TRACKED_WEEKLY_NUTRITION_FIELDS = [
    "calorie",
    "net_carbs",
    "gl",
    "fat",
    "saturated_fat",
    "cholesterol",
    "sodium",
    "salt_equivalent",
    "actual_weight"
]


def is_ark_vision_configured():
    return bool(ARK_API_KEY and ARK_VISION_MODEL_ENDPOINT and ARK_VISION_URL)


def is_ark_text_configured():
    return bool(ARK_API_KEY and ARK_TEXT_MODEL_ENDPOINT and ARK_CHAT_URL)

# ==================== 全局配置数据 ====================
# 默认健康参数（前端未提交用户数据时，使用该默认值计算饮食建议）
default_health_params = {
    "daily_net_carbs": 80.0,
    "fasting_sugar": 6.5,
    "daily_fat": 50.0,
    "daily_saturated_fat": 15.0,
    "daily_sodium": 2000.0,
    "blood_pressure": 140,
    "daily_energy": 1600.0
}

food_db = {}
model = None

# 菜品 ID 与名称映射表（YOLO 模型输出 ID -> 对应食物中文名称）
id2name = {
    0: "麻婆豆腐", 1: "家常豆腐", 2: "煎豆腐", 3: "豆腐花", 4: "臭豆腐",
    5: "酸辣土豆丝", 6: "土豆泥", 7: "香煎土豆", 8: "土豆焖豆角", 9: "地三鲜",
    10: "薯条", 11: "鱼香茄子", 12: "蒜泥茄子", 13: "肉末茄子", 14: "辣白菜",
    15: "醋溜白菜", 16: "上汤娃娃菜", 17: "手撕包菜", 18: "蚝油生菜", 19: "炒青菜",
    20: "炒空心菜", 21: "蒜蓉油麦菜", 22: "清炒菠菜", 23: "炒豆芽", 24: "炒蚕豆",
    25: "毛豆", 26: "蚝油西兰花", 27: "香煎藕盒", 28: "莲藕", 29: "凉拌西红柿",
    30: "鸡鸭胗", 31: "凉拌木耳", 32: "口水黄瓜", 33: "花生米", 34: "凉拌海带丝",
    35: "拔丝山药", 36: "清炒山药", 37: "干煸豆角", 38: "蚝油杏鲍菇", 39: "酿苦瓜",
    40: "炒苦瓜", 41: "虎皮青椒", 42: "凉拌腐竹", 43: "炒花菜", 44: "松仁玉米",
    45: "香菇青菜", 46: "椒盐蘑菇", 47: "芹菜香干", 48: "西芹百合", 49: "韭菜炒香干",
    50: "西红柿炒鸡蛋", 51: "韭菜炒鸡蛋", 52: "黄瓜炒鸡蛋", 53: "鸡蛋羹", 54: "猪肝",
    55: "猪耳朵", 56: "叉烧", 57: "粉蒸排骨", 58: "糖醋排骨", 59: "海带炖排骨",
    60: "可乐鸡翅", 61: "泡椒凤爪", 62: "红烧鸡爪", 63: "口水鸡", 64: "烤鸭烧鹅",
    65: "白斩鸡", 66: "大盘鸡", 67: "香菇蒸鸡", 68: "黄焖鸡", 69: "豉油鸡",
    70: "辣子鸡", 71: "宫保鸡丁", 72: "三杯鸡", 73: "鸡丝、鸡丝面", 74: "炸鸡腿",
    75: "啤酒鸭", 76: "腰花", 77: "红烧肉", 78: "红烧牛肉", 79: "酱牛肉",
    80: "西红柿牛腩", 81: "土豆炖牛腩", 82: "杭椒牛柳", 83: "梅菜扣肉", 84: "回锅肉",
    85: "猪肉炖粉条", 86: "水煮肉片", 87: "糖醋里脊", 88: "咕噜肉", 89: "锅包肉",
    90: "农家小炒肉", 91: "培根金针菇卷", 92: "京酱肉丝", 93: "豆角肉丝", 94: "酱焖猪蹄",
    95: "肚丝", 96: "青椒肉丝", 97: "鱼香肉丝", 98: "木耳炒肉丝", 99: "木须肉",
    100: "莴笋肉丝", 101: "蚂蚁上树", 102: "孜然羊肉", 103: "羊肉串", 104: "葱爆羊肉",
    105: "红烧狮子头", 106: "酸菜鱼", 107: "烤鱼", 108: "糖醋鲤鱼", 109: "松鼠桂鱼",
    110: "红烧带鱼", 111: "剁椒鱼头", 112: "水煮鱼", 113: "清蒸鲈鱼", 114: "芝士虾球",
    115: "虾仁西兰花", 116: "油焖大虾", 117: "香辣虾", 118: "香辣小龙虾", 119: "水晶虾饺",
    120: "蒜茸粉丝蒸虾", 121: "清炒虾仁", 122: "皮皮虾", 123: "扇贝", 124: "生蚝",
    125: "鱿鱼", 126: "鲍鱼", 127: "螃蟹", 128: "甲鱼", 129: "鳝鱼", 130: "扬州炒饭",
    131: "蛋包饭", 132: "小笼汤包", 133: "烧麦", 134: "家常早餐鸡蛋饼", 135: "土豆鸡蛋饼",
    136: "鸡蛋灌饼", 137: "卤蛋", 138: "荷包蛋", 139: "葱花手抓饼", 140: "芝麻烧饼",
    141: "肉夹馍", 142: "韭菜盒子", 143: "南瓜紫薯馒头", 144: "馒头", 145: "包子",
    146: "南瓜饼", 147: "披萨", 148: "油条", 149: "炸酱面", 150: "重庆酸辣粉",
    151: "凉拌凉面", 152: "西红柿鸡蛋面", 153: "肉酱意大利面", 154: "茄汁拌面",
    155: "凉皮", 156: "担担面", 157: "臊子面", 158: "炒面", 159: "饺子",
    160: "玉米棒", 161: "红烧牛肉面", 162: "河粉", 163: "肠粉", 164: "鲜肉小馄饨",
    165: "煎饺", 166: "汤圆", 167: "小米粥", 168: "红薯粥", 169: "海蛰",
    170: "皮蛋瘦肉粥", 171: "大米粥", 172: "米饭", 173: "紫菜包饭", 174: "石锅饭",
    175: "乌鸡汤", 176: "鲫鱼豆腐汤", 177: "疙瘩汤", 178: "酸辣汤", 179: "萝卜排骨汤",
    180: "西红柿鸡蛋汤", 181: "西湖牛肉羹", 182: "莲藕排骨汤", 183: "紫菜蛋花汤",
    184: "海带豆腐汤", 185: "玉米排骨汤", 186: "菠菜猪肝汤", 187: "罗宋汤",
    188: "银耳汤", 189: "冬瓜汤", 190: "酱汤", 191: "毛血旺", 192: "夫妻肺片",
    193: "麻辣香锅", 194: "黄金如意肉卷", 195: "蛋糕", 196: "蛋挞", 197: "面包",
    198: "牛角包", 199: "吐司", 200: "饼干", 201: "曲奇饼干", 202: "苏打饼干",
    203: "双皮奶", 204: "冰激凌", 205: "鸡蛋布丁", 206: "冰糖雪梨", 207: "水果沙拉"
}


def safe_float(value, default=0.0):
    try:
        if value in ("", None, "---"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value in ("", None, "---"):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def now_local():
    return datetime.now()


def now_iso():
    return now_local().strftime("%Y-%m-%d %H:%M:%S")


def parse_datetime(value):
    if not value:
        return None

    text = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def ratio_to_percent(value):
    return round(safe_float(value) * 100, 2)


def get_disease_detail_key(disease):
    return {
        "diabetes": "diabetes_detail",
        "hyperglycemia": "hyperglycemia_detail",
        "hyperlipidemia": "hyperlipidemia_detail",
        "hypertension": "hypertension_detail"
    }.get(str(disease or "").strip().lower(), "")


def normalize_activity_level(value):
    text = str(value or "").strip().lower()
    if text in {"low", "light", "sedentary"}:
        return "light"
    if text in {"medium", "moderate", "mid"}:
        return "medium"
    if text in {"high", "heavy", "active"}:
        return "high"
    if text in {"bed", "bedridden"}:
        return "bed"
    return "light"


def normalize_yes_no_bool(value, default=False):
    if value in ("yes", "y", "1", 1, True):
        return True
    if value in ("no", "n", "0", 0, False):
        return False
    return parse_bool(value, default)


def infer_has_ckd_from_draft(other, detail, disease):
    other = dict(other or {})
    detail = dict(detail or {})
    disease = str(disease or "").strip().lower()

    egfr = safe_float(other.get("egfr"))
    if 0 < egfr < 60:
        return True

    if disease in {"diabetes", "hyperglycemia"}:
        if str(detail.get("diabetic_nephropathy", "")).strip().lower() in {"suspected", "confirmed"}:
            return True
        if str(detail.get("proteinuria", "")).strip().lower() in {"micro", "overt"}:
            return True

    if disease == "hypertension":
        comorbidities = {str(item).strip().lower() for item in (detail.get("comorbidities") or [])}
        if "ckd" in comorbidities:
            return True

    return False


def standardize_health_profile_from_draft(base_info=None, disease_detail=None, disease=""):
    base_info = dict(base_info or {})
    basic = dict(base_info.get("basic") or {})
    other = dict(base_info.get("other") or {})
    conditions = list(base_info.get("conditions") or [])
    detail = dict(disease_detail or {})
    disease = str(disease or "").strip().lower()

    fasting_sugar = default_health_params["fasting_sugar"]
    if disease in {"diabetes", "hyperglycemia"}:
        fasting_sugar = safe_float(detail.get("fasting_glucose"), default_health_params["fasting_sugar"])

    standardized = {
        "activity_level": normalize_activity_level(basic.get("activity")),
        "height_cm": safe_float(basic.get("height")),
        "weight_kg": safe_float(basic.get("weight")),
        "fasting_sugar": fasting_sugar,
        "has_ckd": infer_has_ckd_from_draft(other, detail, disease),
        "ascvd_history": normalize_yes_no_bool(detail.get("ascvd_history"), False),
        "disease": disease,
        "conditions": conditions,
        "raw_base_info": base_info,
        "raw_disease_detail": detail
    }

    if "egfr" in other:
        standardized["egfr"] = safe_float(other.get("egfr"))
    if "age" in basic:
        standardized["age"] = safe_int(basic.get("age"))
    if "gender" in basic:
        standardized["gender"] = str(basic.get("gender") or "").strip().lower()
    if "waist" in basic:
        standardized["waist_cm"] = safe_float(basic.get("waist"))
    if "pregnancy_status" in basic:
        standardized["pregnancy_status"] = str(basic.get("pregnancy_status") or "").strip().lower()
    if "allergies" in other:
        standardized["allergies"] = list(other.get("allergies") or [])

    return standardized


def build_request_health_profile(payload, disease=""):
    payload = dict(payload or {})
    existing_profile = dict(payload.get("health_profile") or {})
    disease = str(
        disease
        or payload.get("disease")
        or existing_profile.get("disease")
        or ""
    ).strip().lower()

    base_info = {}
    if any(key in payload for key in ("basic", "other", "conditions")):
        base_info = {
            "basic": dict(payload.get("basic") or {}),
            "other": dict(payload.get("other") or {}),
            "conditions": list(payload.get("conditions") or [])
        }

    detail_key = get_disease_detail_key(disease)
    disease_detail = dict(payload.get(detail_key) or {}) if detail_key else {}
    standardized_from_draft = standardize_health_profile_from_draft(base_info, disease_detail, disease) if base_info else {}

    if standardized_from_draft:
        merged = dict(existing_profile)
        for key, value in standardized_from_draft.items():
            if key in {"raw_base_info", "raw_disease_detail", "conditions", "allergies"}:
                merged[key] = value
            elif key not in merged or merged.get(key) in ("", None, [], {}):
                merged[key] = value
        merged["disease"] = disease or merged.get("disease") or standardized_from_draft.get("disease", "")
        return merged

    if existing_profile:
        existing_profile.setdefault("disease", disease or str(existing_profile.get("disease") or "").strip().lower())
        return existing_profile

    return {"disease": disease} if disease else {}


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    ensure_data_dir()
    with get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS diet_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recognition_key TEXT NOT NULL,
                disease TEXT NOT NULL,
                food_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                level TEXT,
                portion_text TEXT,
                tip TEXT,
                visual_tip TEXT,
                actual_weight REAL,
                percentage INTEGER,
                recommended_grams REAL,
                meal_slot TEXT,
                risk_ratio REAL,
                main_risk_reason TEXT,
                nutrition_json TEXT NOT NULL,
                analysis_json TEXT,
                health_profile_json TEXT,
                source_payload_json TEXT,
                captured_at TEXT,
                recorded_at TEXT NOT NULL,
                canceled_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, recognition_key)
            );

            CREATE INDEX IF NOT EXISTS idx_diet_records_user_time
            ON diet_records(user_id, recorded_at DESC);

            CREATE INDEX IF NOT EXISTS idx_diet_records_status
            ON diet_records(status);

            CREATE INDEX IF NOT EXISTS idx_diet_records_disease
            ON diet_records(disease);
            """
        )


def json_dumps(data):
    return json.dumps(data or {}, ensure_ascii=False)


def json_loads(text, default=None):
    if not text:
        return {} if default is None else default
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {} if default is None else default


def resolve_user_id(payload=None):
    payload = payload or {}
    user_id = (
        payload.get("user_id")
        or payload.get("openid")
        or request.args.get("user_id", "")
        or request.args.get("openid", "")
    )
    user_id = str(user_id).strip()
    return user_id or DEFAULT_USER_ID


def infer_meal_slot(meal_slot=None, recorded_at=None):
    if meal_slot:
        meal_slot = str(meal_slot).strip().lower()
        if meal_slot in MEAL_FACTORS:
            return meal_slot

    dt = parse_datetime(recorded_at) or now_local()
    hour = dt.hour
    if 5 <= hour < 10:
        return "breakfast"
    if 10 <= hour < 15:
        return "lunch"
    if 15 <= hour < 21:
        return "dinner"
    return "snack"


def get_meal_factor(meal_slot):
    return MEAL_FACTORS.get(infer_meal_slot(meal_slot), 0.30)


def load_food_database():
    global food_db

    if food_db:
        return

    print("\n🔄 加载营养数据库...")
    content = (BASE_DIR / "food_nutrition.json").read_text(encoding="utf-8")
    decoder = JSONDecoder()
    pos = 0
    content_len = len(content)
    loaded_food_db = {}

    while pos < content_len:
        try:
            obj, new_pos = decoder.raw_decode(content, pos)
            if isinstance(obj, dict) and obj:
                loaded_food_db.update(obj)
            pos = new_pos
        except json.JSONDecodeError:
            pos += 1

    food_db = loaded_food_db
    print(f"✅ 营养数据库加载完成，共 {len(food_db)} 种食物")


def load_model():
    global model

    if model is not None:
        return

    if YOLO is None:
        raise RuntimeError("ultralytics 未安装，无法加载识别模型")

    print("\n🔄 加载 YOLO 模型...")
    model = YOLO(str(BASE_DIR / "model" / "best.pt"))
    print("✅ 模型加载成功")


def build_ark_prompt(label):
    return (
        "我已经用本地自研模型识别出图片盘子里的核心食物是'{}'。"
        "请你作为一个视觉空间专家，仔细分析图片，帮我估算出这个'{}'在盘子/碗里的面积/体积占比百分比"
        "（返回1-100的整数）。同时，请根据你估算出的比例和日常生活常识，用极其直观、接地气、适合慢病患者阅读的日常语言，"
        "描述这个分量大概是多少，例如：'约半碗米饭'、'相当于3满勺'、'一小盘的量'、'两小块'。"
        "必须严格按照以下 JSON 格式返回，不要包含任何前后解释文字或 markdown 标记：\n"
        "{{\n"
        '  "percentage": 45,\n'
        '  "visual_tip": "这里是大模型生成的直观分量描述文本"\n'
        "}}"
    ).format(label, label)


def extract_ark_output_text(response_json):
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "").strip()
    return ""


def parse_percentage_value(raw_text):
    cleaned_text = (raw_text or "").strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`").replace("json", "", 1).strip()

    json_start = cleaned_text.find("{")
    json_end = cleaned_text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end >= json_start:
        cleaned_text = cleaned_text[json_start:json_end + 1]

    parsed_data = json.loads(cleaned_text)
    percentage = safe_int(parsed_data.get("percentage", 50), 50)
    visual_tip = parsed_data.get("visual_tip", DEFAULT_VISUAL_TIP)
    if not isinstance(visual_tip, str) or not visual_tip.strip():
        visual_tip = DEFAULT_VISUAL_TIP

    return max(1, min(100, percentage)), visual_tip.strip()


def estimate_food_percentage(img_base64, label):
    if requests is None:
        print("requests 未安装，体积估算默认回退到 50%")
        return 50, DEFAULT_VISUAL_TIP

    if not is_ark_vision_configured():
        print("ARK vision config missing, fallback to default 50% portion estimate")
        return 50, DEFAULT_VISUAL_TIP

    prompt = build_ark_prompt(label)
    payload = {
        "model": ARK_VISION_MODEL_ENDPOINT,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img_base64}"
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            ARK_VISION_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=ARK_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        response_json = response.json()
        output_text = extract_ark_output_text(response_json)
        return parse_percentage_value(output_text)
    except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"豆包体积估算失败: {exc}")
        return 50, DEFAULT_VISUAL_TIP


def get_standard_weight(food_nutri):
    standard_weight = safe_float(food_nutri.get("standard_weight_g"), 0)
    return standard_weight if standard_weight > 0 else DEFAULT_FULL_PORTION_GRAMS


def build_actual_nutrition(food_nutri, percentage):
    standard_weight = get_standard_weight(food_nutri)
    actual_weight = round(standard_weight * (percentage / 100.0), 2)
    weight_factor = actual_weight / 100.0
    scaled_nutri = {}

    for key, value in food_nutri.items():
        if key in SCALABLE_NUTRITION_FIELDS and isinstance(value, (int, float)):
            scaled_nutri[key] = round(value * weight_factor, 2)
        else:
            scaled_nutri[key] = value

    # GI 是食物属性，不随重量变化
    if isinstance(food_nutri.get("gi"), (int, float)):
        scaled_nutri["gi"] = safe_float(food_nutri.get("gi"))

    scaled_nutri["standard_weight_g"] = round(standard_weight, 2)
    scaled_nutri["actual_weight"] = actual_weight
    scaled_nutri["percentage"] = percentage
    return scaled_nutri


def format_nutrition_value(value):
    if isinstance(value, (int, float)):
        rounded_value = round(value, 2)
        if float(rounded_value).is_integer():
            return str(int(rounded_value))
        return f"{rounded_value:.2f}".rstrip("0").rstrip(".")
    return str(value) if value is not None else "---"


def normalize_health_profile(health_profile, disease):
    health_profile = dict(health_profile or {})
    disease = str(disease or health_profile.get("disease") or "").strip().lower()

    if "basic" in health_profile or "other" in health_profile:
        detail_key = get_disease_detail_key(disease)
        detail = dict(health_profile.get(detail_key) or {}) if detail_key else {}
        health_profile = standardize_health_profile_from_draft(health_profile, detail, disease)

    activity = normalize_activity_level(health_profile.get("activity_level"))
    height_cm = safe_float(health_profile.get("height_cm"))
    weight_kg = safe_float(health_profile.get("weight_kg"))
    fasting_sugar = safe_float(health_profile.get("fasting_sugar"), default_health_params["fasting_sugar"])

    profile = {
        "activity_level": activity or "light",
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "fasting_sugar": fasting_sugar,
        "has_ckd": parse_bool(health_profile.get("has_ckd") or health_profile.get("kidney_disease"), False),
        "ascvd_history": parse_bool(health_profile.get("ascvd_history"), False),
        "disease": disease
    }

    height_m = height_cm / 100 if height_cm > 0 else 0
    bmi = weight_kg / (height_m ** 2) if height_m > 0 and weight_kg > 0 else 0
    profile["bmi"] = round(bmi, 2) if bmi > 0 else 0

    if height_m > 0:
        ideal_weight = 22 * (height_m ** 2)
    else:
        ideal_weight = weight_kg or 0

    if weight_kg > 0:
        if bmi >= 24:
            adjusted_weight = ideal_weight + 0.5 * (weight_kg - ideal_weight)
        else:
            adjusted_weight = weight_kg
    else:
        adjusted_weight = safe_float(health_profile.get("adjusted_weight"), 0)

    profile["ideal_weight"] = round(ideal_weight, 2) if ideal_weight > 0 else 0
    profile["adjusted_weight"] = round(adjusted_weight, 2) if adjusted_weight > 0 else 0

    energy = safe_float(health_profile.get("daily_energy"), 0)
    if energy <= 0:
        if adjusted_weight > 0:
            if bmi >= 28:
                base_factor = 20
            elif bmi >= 24:
                base_factor = 25
            else:
                base_factor = 30

            if activity in {"medium", "moderate"}:
                base_factor += 5
            elif activity in {"heavy", "high"}:
                base_factor += 10
            elif activity in {"bed", "bedridden"}:
                base_factor = max(15, base_factor - 5)

            energy = adjusted_weight * base_factor
        else:
            energy = default_health_params["daily_energy"]

    profile["daily_energy"] = round(energy, 2)
    return profile


def derive_daily_limits(profile, disease):
    daily_energy = safe_float(profile.get("daily_energy"), default_health_params["daily_energy"])
    fasting_sugar = safe_float(profile.get("fasting_sugar"), default_health_params["fasting_sugar"])
    has_ckd = parse_bool(profile.get("has_ckd"), False)
    ascvd_history = parse_bool(profile.get("ascvd_history"), False)

    daily_limits = {
        "daily_energy": round(daily_energy, 2),
        "daily_net_carbs": default_health_params["daily_net_carbs"],
        "daily_fat": default_health_params["daily_fat"],
        "daily_saturated_fat": default_health_params["daily_saturated_fat"],
        "daily_sodium": default_health_params["daily_sodium"],
        "daily_cholesterol": 200.0 if ascvd_history else 300.0,
        "fasting_sugar": fasting_sugar
    }

    if disease in {"diabetes", "hyperglycemia"}:
        daily_limits["daily_net_carbs"] = round(daily_energy * 0.50 / 4.0, 2)

    if disease == "hyperlipidemia":
        daily_limits["daily_fat"] = round(daily_energy * 0.25 / 9.0, 2)
        daily_limits["daily_saturated_fat"] = round(daily_energy * 0.07 / 9.0, 2)

    if disease == "hypertension" or has_ckd:
        daily_limits["daily_sodium"] = 1500.0
    else:
        daily_limits["daily_sodium"] = safe_float(
            profile.get("daily_sodium"),
            default_health_params["daily_sodium"]
        )

    return daily_limits


def calculate_meal_limits(disease, health_profile, meal_slot):
    profile = normalize_health_profile(health_profile, disease)
    daily_limits = derive_daily_limits(profile, disease)
    meal_factor = get_meal_factor(meal_slot)

    meal_limits = {
        "meal_slot": infer_meal_slot(meal_slot),
        "meal_factor": round(meal_factor, 2),
        "daily_limits": daily_limits
    }

    if disease in {"diabetes", "hyperglycemia"}:
        meal_limits["net_carbs"] = round(daily_limits["daily_net_carbs"] * meal_factor, 2)
        meal_limits["gl"] = 20.0

    elif disease == "hyperlipidemia":
        meal_limits["fat"] = round(daily_limits["daily_fat"] * meal_factor, 2)
        meal_limits["saturated_fat"] = round(daily_limits["daily_saturated_fat"] * meal_factor, 2)
        meal_limits["cholesterol"] = round(daily_limits["daily_cholesterol"] * meal_factor, 2)

    elif disease == "hypertension":
        meal_limits["sodium"] = round(daily_limits["daily_sodium"] * meal_factor, 2)
        meal_limits["salt_equivalent"] = round((daily_limits["daily_sodium"] / 1000.0) * 2.5 * meal_factor, 2)

    else:
        meal_limits["net_carbs"] = round(daily_limits["daily_net_carbs"] * meal_factor, 2)
        meal_limits["gl"] = 20.0

    return meal_limits


def build_ratio_item(label, nutrient_key, actual_value, limit_value):
    actual_num = safe_float(actual_value)
    limit_num = safe_float(limit_value)
    ratio = actual_num / limit_num if limit_num > 0 else 0
    return {
        "label": label,
        "nutrient_key": nutrient_key,
        "actual_value": round(actual_num, 2),
        "limit_value": round(limit_num, 2),
        "ratio": round(ratio, 4),
        "percent": ratio_to_percent(ratio)
    }


def calculate_indicator_ratios(food_nutri, actual_nutri, disease, meal_limits):
    ratio_items = []

    if disease in {"diabetes", "hyperglycemia"}:
        ratio_items.append(
            build_ratio_item("净碳水占本餐上限", "net_carbs", actual_nutri.get("net_carbs"), meal_limits.get("net_carbs"))
        )
        ratio_items.append(
            build_ratio_item("GL 占本餐参考上限", "gl", actual_nutri.get("gl"), meal_limits.get("gl"))
        )

    elif disease == "hyperlipidemia":
        ratio_items.append(
            build_ratio_item("脂肪占本餐上限", "fat", actual_nutri.get("fat"), meal_limits.get("fat"))
        )
        ratio_items.append(
            build_ratio_item("饱和脂肪占本餐上限", "saturated_fat", actual_nutri.get("saturated_fat"), meal_limits.get("saturated_fat"))
        )
        ratio_items.append(
            build_ratio_item("胆固醇占本餐上限", "cholesterol", actual_nutri.get("cholesterol"), meal_limits.get("cholesterol"))
        )

    elif disease == "hypertension":
        ratio_items.append(
            build_ratio_item("钠占本餐上限", "sodium", actual_nutri.get("sodium"), meal_limits.get("sodium"))
        )
        ratio_items.append(
            build_ratio_item("食盐当量占本餐上限", "salt_equivalent", actual_nutri.get("salt_equivalent"), meal_limits.get("salt_equivalent"))
        )

    else:
        ratio_items.append(
            build_ratio_item("净碳水占本餐上限", "net_carbs", actual_nutri.get("net_carbs"), meal_limits.get("net_carbs"))
        )
        ratio_items.append(
            build_ratio_item("GL 占本餐参考上限", "gl", actual_nutri.get("gl"), meal_limits.get("gl"))
        )

    ratio_items = [item for item in ratio_items if item["limit_value"] > 0]
    risk_ratio = max((item["ratio"] for item in ratio_items), default=0)
    return ratio_items, round(risk_ratio, 4)


def base_level_from_ratio(risk_ratio):
    if risk_ratio <= 0.30:
        return "SAFE"
    if risk_ratio <= 0.70:
        return "CAUTION"
    return "AVOID"


def upgrade_level(level):
    if level == "SAFE":
        return "CAUTION"
    if level == "CAUTION":
        return "AVOID"
    return "AVOID"


def apply_hard_label(level, food_nutri, actual_nutri, disease):
    if disease in {"diabetes", "hyperglycemia"}:
        high_gi = safe_float(food_nutri.get("gi")) >= 70
        high_gl = safe_float(actual_nutri.get("gl")) >= 20
        if high_gi and high_gl:
            return upgrade_level(level)

    if disease == "hypertension":
        if safe_float(food_nutri.get("sodium")) >= 600:
            return upgrade_level(level)

    if disease == "hyperlipidemia":
        if safe_float(food_nutri.get("saturated_fat")) >= 5:
            return upgrade_level(level)

    return level


def calculate_recommended_grams(food_nutri, disease, meal_limits, target_ratio=RECOMMENDED_RATIO_TARGET):
    standard_weight = get_standard_weight(food_nutri)
    allow_values = [standard_weight]

    def append_allow(limit_key, nutrient_key):
        limit_value = safe_float(meal_limits.get(limit_key))
        nutrient_per_100g = safe_float(food_nutri.get(nutrient_key))
        if limit_value > 0 and nutrient_per_100g > 0:
            allow = limit_value * target_ratio * 100.0 / nutrient_per_100g
            allow_values.append(allow)

    if disease in {"diabetes", "hyperglycemia"}:
        append_allow("net_carbs", "net_carbs")
        append_allow("gl", "gl")

    elif disease == "hyperlipidemia":
        append_allow("fat", "fat")
        append_allow("saturated_fat", "saturated_fat")
        append_allow("cholesterol", "cholesterol")

    elif disease == "hypertension":
        append_allow("sodium", "sodium")
        append_allow("salt_equivalent", "salt_equivalent")

    else:
        append_allow("net_carbs", "net_carbs")
        append_allow("gl", "gl")

    recommended_grams = max(1.0, min(allow_values))
    return round(recommended_grams, 2)


def build_main_risk_reason(ratio_items):
    if not ratio_items:
        return "暂无明显风险项"
    top_item = max(ratio_items, key=lambda item: item["ratio"])
    if top_item["ratio"] <= 0:
        return "暂无明显风险项"
    return top_item["label"]


def select_tip(food_nutri, disease):
    if disease in {"diabetes", "hyperglycemia"}:
        return food_nutri.get("tip1", "请结合总量控制食用。")
    if disease == "hyperlipidemia":
        return food_nutri.get("tip2", "请结合脂肪与胆固醇控制食用。")
    if disease == "hypertension":
        return food_nutri.get("tip3", "请结合钠与盐控制食用。")
    return food_nutri.get("tip1", "请结合健康目标控制食用。")


def generate_recognition_key():
    return f"rec_{uuid.uuid4().hex}"


def build_analysis_result(label, food_nutri, disease, img_base64, health_profile=None, meal_slot=None):
    percentage, visual_tip = estimate_food_percentage(img_base64, label)
    actual_nutri = build_actual_nutrition(food_nutri, percentage)
    meal_slot = infer_meal_slot(meal_slot)
    meal_limits = calculate_meal_limits(disease, health_profile, meal_slot)
    ratio_items, risk_ratio = calculate_indicator_ratios(food_nutri, actual_nutri, disease, meal_limits)
    level = base_level_from_ratio(risk_ratio)
    level = apply_hard_label(level, food_nutri, actual_nutri, disease)
    recommended_grams = calculate_recommended_grams(food_nutri, disease, meal_limits)
    tip = select_tip(food_nutri, disease)
    main_risk_reason = build_main_risk_reason(ratio_items)
    captured_at = now_iso()
    recognition_key = generate_recognition_key()

    nutrition_payload = {
        "calorie": safe_float(actual_nutri.get("calorie")),
        "total_carbs": safe_float(actual_nutri.get("total_carbs")),
        "protein": safe_float(actual_nutri.get("protein")),
        "fat": safe_float(actual_nutri.get("fat")),
        "net_carbs": safe_float(actual_nutri.get("net_carbs")),
        "dietary_fiber": safe_float(actual_nutri.get("dietary_fiber")),
        "gl": safe_float(actual_nutri.get("gl")),
        "gi": safe_float(actual_nutri.get("gi")),
        "sodium": safe_float(actual_nutri.get("sodium")),
        "added_sugar": safe_float(actual_nutri.get("added_sugar")),
        "saturated_fat": safe_float(actual_nutri.get("saturated_fat")),
        "cholesterol": safe_float(actual_nutri.get("cholesterol")),
        "salt_equivalent": safe_float(actual_nutri.get("salt_equivalent")),
        "actual_weight": safe_float(actual_nutri.get("actual_weight")),
        "percentage": safe_int(actual_nutri.get("percentage")),
        "standard_weight_g": safe_float(actual_nutri.get("standard_weight_g"))
    }

    analysis_payload = {
        "recognition_key": recognition_key,
        "captured_at": captured_at,
        "disease": disease,
        "meal_slot": meal_slot,
        "level": level,
        "risk_ratio": risk_ratio,
        "risk_ratio_percent": ratio_to_percent(risk_ratio),
        "main_risk_reason": main_risk_reason,
        "meal_limits": meal_limits,
        "meal_limit_ratios": ratio_items,
        "recommended_grams": recommended_grams,
        "target_ratio": RECOMMENDED_RATIO_TARGET,
        "health_profile": normalize_health_profile(health_profile, disease)
    }

    portion_text = f"建议吃 {int(round(recommended_grams))} 克"

    return {
        "imgBase64": img_base64,
        "food": label,
        "disease": disease,
        "level": level,
        "portion": portion_text,
        "tip": tip,
        "visual_tip": visual_tip,
        "recognition_key": recognition_key,
        "captured_at": captured_at,
        "meal_slot": meal_slot,
        "actual_weight": actual_nutri["actual_weight"],
        "percentage": actual_nutri["percentage"],
        "recommendation_grams": recommended_grams,
        "risk_ratio": risk_ratio,
        "risk_ratio_percent": ratio_to_percent(risk_ratio),
        "main_risk_reason": main_risk_reason,
        "cal": format_nutrition_value(nutrition_payload["calorie"]),
        "net_carbs": format_nutrition_value(nutrition_payload["net_carbs"]),
        "gl": format_nutrition_value(nutrition_payload["gl"]),
        "gi": format_nutrition_value(nutrition_payload["gi"]),
        "fat": format_nutrition_value(nutrition_payload["fat"]),
        "total_fat": format_nutrition_value(nutrition_payload["fat"]),
        "saturated_fat": format_nutrition_value(nutrition_payload["saturated_fat"]),
        "cholesterol": format_nutrition_value(nutrition_payload["cholesterol"]),
        "sodium": format_nutrition_value(nutrition_payload["sodium"]),
        "salt_equivalent": format_nutrition_value(nutrition_payload["salt_equivalent"]),
        "saltEquivalent": format_nutrition_value(nutrition_payload["salt_equivalent"]),
        "salt": format_nutrition_value(nutrition_payload["salt_equivalent"]),
        "nutrition": nutrition_payload,
        "analysis": analysis_payload
    }


def empty_detect_response(img_base64, food, level, portion, tip):
    return {
        "imgBase64": img_base64,
        "food": food,
        "level": level,
        "portion": portion,
        "cal": "---",
        "net_carbs": "---",
        "gl": "---",
        "gi": "---",
        "fat": "---",
        "total_fat": "---",
        "saturated_fat": "---",
        "cholesterol": "---",
        "sodium": "---",
        "salt_equivalent": "---",
        "saltEquivalent": "---",
        "salt": "---",
        "tip": tip,
        "visual_tip": DEFAULT_VISUAL_TIP,
        "recognition_key": generate_recognition_key(),
        "captured_at": now_iso(),
        "meal_slot": infer_meal_slot(),
        "actual_weight": 0,
        "percentage": 0,
        "recommendation_grams": 0,
        "risk_ratio": 0,
        "risk_ratio_percent": 0,
        "main_risk_reason": "暂无",
        "nutrition": {},
        "analysis": {}
    }


def normalize_recognition_result(payload):
    recognition = dict(payload.get("recognition_result") or payload.get("detection_result") or payload)
    nutrition = recognition.get("nutrition") or {}
    analysis = recognition.get("analysis") or {}
    disease = recognition.get("disease") or payload.get("disease") or ""

    normalized = {
        "recognition_key": recognition.get("recognition_key") or payload.get("recognition_key") or generate_recognition_key(),
        "food": recognition.get("food") or payload.get("food") or "",
        "disease": disease,
        "level": recognition.get("level") or payload.get("level") or "",
        "portion": recognition.get("portion") or payload.get("portion") or "",
        "tip": recognition.get("tip") or payload.get("tip") or "",
        "visual_tip": recognition.get("visual_tip") or payload.get("visual_tip") or DEFAULT_VISUAL_TIP,
        "actual_weight": safe_float(
            recognition.get("actual_weight", nutrition.get("actual_weight", payload.get("actual_weight")))
        ),
        "percentage": safe_int(recognition.get("percentage", nutrition.get("percentage", payload.get("percentage")))),
        "recommendation_grams": safe_float(
            recognition.get("recommendation_grams", analysis.get("recommended_grams", payload.get("recommendation_grams")))
        ),
        "risk_ratio": safe_float(recognition.get("risk_ratio", analysis.get("risk_ratio", payload.get("risk_ratio")))),
        "main_risk_reason": recognition.get("main_risk_reason") or analysis.get("main_risk_reason") or payload.get("main_risk_reason") or "",
        "meal_slot": infer_meal_slot(recognition.get("meal_slot") or payload.get("meal_slot"), recognition.get("captured_at")),
        "captured_at": recognition.get("captured_at") or payload.get("captured_at") or now_iso(),
        "nutrition": nutrition if nutrition else {
            "calorie": safe_float(recognition.get("cal")),
            "net_carbs": safe_float(recognition.get("net_carbs")),
            "gl": safe_float(recognition.get("gl")),
            "gi": safe_float(recognition.get("gi")),
            "fat": safe_float(recognition.get("fat") or recognition.get("total_fat")),
            "saturated_fat": safe_float(recognition.get("saturated_fat")),
            "cholesterol": safe_float(recognition.get("cholesterol")),
            "sodium": safe_float(recognition.get("sodium")),
            "salt_equivalent": safe_float(recognition.get("salt_equivalent") or recognition.get("salt")),
            "actual_weight": safe_float(recognition.get("actual_weight")),
            "percentage": safe_int(recognition.get("percentage"))
        },
        "analysis": analysis,
        "source_payload": recognition
    }
    return normalized


def serialize_record(row):
    nutrition = json_loads(row["nutrition_json"])
    analysis = json_loads(row["analysis_json"])
    health_profile = json_loads(row["health_profile_json"])
    source_payload = json_loads(row["source_payload_json"])

    return {
        "record_id": row["id"],
        "user_id": row["user_id"],
        "recognition_key": row["recognition_key"],
        "disease": row["disease"],
        "food": row["food_name"],
        "status": row["status"],
        "level": row["level"],
        "portion": row["portion_text"],
        "tip": row["tip"],
        "visual_tip": row["visual_tip"],
        "actual_weight": safe_float(row["actual_weight"]),
        "percentage": safe_int(row["percentage"]),
        "recommendation_grams": safe_float(row["recommended_grams"]),
        "meal_slot": row["meal_slot"],
        "risk_ratio": safe_float(row["risk_ratio"]),
        "risk_ratio_percent": ratio_to_percent(row["risk_ratio"]),
        "main_risk_reason": row["main_risk_reason"] or "",
        "captured_at": row["captured_at"],
        "recorded_at": row["recorded_at"],
        "canceled_at": row["canceled_at"],
        "nutrition": nutrition,
        "analysis": analysis,
        "health_profile": health_profile,
        "source_payload": source_payload
    }


def get_history_filters():
    disease = request.args.get("disease", "").strip()
    include_canceled = parse_bool(request.args.get("include_canceled"), False)
    page = max(1, safe_int(request.args.get("page"), 1))
    page_size = min(100, max(1, safe_int(request.args.get("page_size"), 20)))
    return disease, include_canceled, page, page_size


def fetch_history_records(user_id, disease="", include_canceled=False, page=1, page_size=20):
    where_parts = ["user_id = ?"]
    params = [user_id]

    if disease:
        where_parts.append("disease = ?")
        params.append(disease)

    if not include_canceled:
        where_parts.append("status = 'active'")

    where_sql = " AND ".join(where_parts)
    offset = (page - 1) * page_size

    with get_db_connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM diet_records WHERE {where_sql}",
            params
        ).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT *
            FROM diet_records
            WHERE {where_sql}
            ORDER BY recorded_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset]
        ).fetchall()

    return total, [serialize_record(row) for row in rows]


def fetch_weekly_records(user_id, disease, days=7, include_canceled=False):
    end_dt = now_local()
    start_dt = end_dt - timedelta(days=max(1, days) - 1)
    start_text = start_dt.strftime("%Y-%m-%d 00:00:00")
    end_text = end_dt.strftime("%Y-%m-%d 23:59:59")

    where_parts = ["user_id = ?", "recorded_at >= ?", "recorded_at <= ?"]
    params = [user_id, start_text, end_text]

    if disease:
        where_parts.append("disease = ?")
        params.append(disease)

    if not include_canceled:
        where_parts.append("status = 'active'")

    with get_db_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM diet_records
            WHERE {' AND '.join(where_parts)}
            ORDER BY recorded_at ASC, id ASC
            """,
            params
        ).fetchall()

    return start_dt, end_dt, [serialize_record(row) for row in rows]


def build_weekly_statistics(records, disease, start_dt, end_dt):
    level_summary = {"SAFE": 0, "CAUTION": 0, "AVOID": 0}
    meal_distribution = {"breakfast": 0, "lunch": 0, "dinner": 0, "snack": 0}
    food_counter = {}
    risk_reason_counter = {}
    nutrient_totals = {field: 0.0 for field in TRACKED_WEEKLY_NUTRITION_FIELDS}
    ratio_bucket = {
        "net_carbs": [],
        "gl": [],
        "fat": [],
        "saturated_fat": [],
        "cholesterol": [],
        "sodium": [],
        "salt_equivalent": []
    }
    daily_totals = {}

    for record in records:
        level = record.get("level", "")
        if level in level_summary:
            level_summary[level] += 1

        meal_slot = infer_meal_slot(record.get("meal_slot"), record.get("recorded_at"))
        meal_distribution[meal_slot] = meal_distribution.get(meal_slot, 0) + 1

        food = record.get("food", "")
        if food:
            food_counter[food] = food_counter.get(food, 0) + 1

        risk_reason = record.get("main_risk_reason", "")
        if risk_reason:
            risk_reason_counter[risk_reason] = risk_reason_counter.get(risk_reason, 0) + 1

        nutrition = record.get("nutrition") or {}
        for field in TRACKED_WEEKLY_NUTRITION_FIELDS:
            nutrient_totals[field] += safe_float(nutrition.get(field))

        for ratio_item in record.get("analysis", {}).get("meal_limit_ratios", []):
            key = ratio_item.get("nutrient_key")
            if key in ratio_bucket:
                ratio_bucket[key].append(safe_float(ratio_item.get("ratio")))

        record_date = (parse_datetime(record.get("recorded_at")) or now_local()).strftime("%Y-%m-%d")
        if record_date not in daily_totals:
            daily_totals[record_date] = {
                "date": record_date,
                "count": 0,
                "safe": 0,
                "caution": 0,
                "avoid": 0,
                "calorie": 0.0,
                "net_carbs": 0.0,
                "fat": 0.0,
                "sodium": 0.0
            }

        day_data = daily_totals[record_date]
        day_data["count"] += 1
        day_data["calorie"] += safe_float(nutrition.get("calorie"))
        day_data["net_carbs"] += safe_float(nutrition.get("net_carbs"))
        day_data["fat"] += safe_float(nutrition.get("fat"))
        day_data["sodium"] += safe_float(nutrition.get("sodium"))
        if level == "SAFE":
            day_data["safe"] += 1
        elif level == "CAUTION":
            day_data["caution"] += 1
        elif level == "AVOID":
            day_data["avoid"] += 1

    total_records = len(records)

    def average_of(values):
        if not values:
            return 0
        return round(sum(values) / len(values), 4)

    def average_nutrition(field):
        if total_records == 0:
            return 0
        return round(nutrient_totals[field] / total_records, 2)

    sorted_foods = sorted(food_counter.items(), key=lambda item: (-item[1], item[0]))
    sorted_reasons = sorted(risk_reason_counter.items(), key=lambda item: (-item[1], item[0]))

    weekly_stats = {
        "period": {
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "days": (end_dt.date() - start_dt.date()).days + 1
        },
        "disease": disease,
        "disease_name": DISEASE_CN_MAP.get(disease, "慢病"),
        "total_records": total_records,
        "level_summary": level_summary,
        "meal_distribution": meal_distribution,
        "top_foods": [{"food": food, "count": count} for food, count in sorted_foods[:5]],
        "top_risk_reasons": [{"reason": reason, "count": count} for reason, count in sorted_reasons[:5]],
        "averages": {
            "calorie": average_nutrition("calorie"),
            "net_carbs": average_nutrition("net_carbs"),
            "gl": average_nutrition("gl"),
            "fat": average_nutrition("fat"),
            "saturated_fat": average_nutrition("saturated_fat"),
            "cholesterol": average_nutrition("cholesterol"),
            "sodium": average_nutrition("sodium"),
            "salt_equivalent": average_nutrition("salt_equivalent"),
            "actual_weight": average_nutrition("actual_weight"),
            "risk_ratio": average_of(
                [safe_float(item.get("risk_ratio")) for item in records]
            ),
            "risk_ratio_percent": ratio_to_percent(
                average_of([safe_float(item.get("risk_ratio")) for item in records])
            )
        },
        "average_limit_ratios": {
            "net_carbs": average_of(ratio_bucket["net_carbs"]),
            "gl": average_of(ratio_bucket["gl"]),
            "fat": average_of(ratio_bucket["fat"]),
            "saturated_fat": average_of(ratio_bucket["saturated_fat"]),
            "cholesterol": average_of(ratio_bucket["cholesterol"]),
            "sodium": average_of(ratio_bucket["sodium"]),
            "salt_equivalent": average_of(ratio_bucket["salt_equivalent"])
        },
        "daily_trends": [
            {
                "date": item["date"],
                "count": item["count"],
                "safe": item["safe"],
                "caution": item["caution"],
                "avoid": item["avoid"],
                "calorie": round(item["calorie"], 2),
                "net_carbs": round(item["net_carbs"], 2),
                "fat": round(item["fat"], 2),
                "sodium": round(item["sodium"], 2)
            }
            for item in sorted(daily_totals.values(), key=lambda value: value["date"])
        ]
    }

    weekly_stats["summary_data"] = {
        "total_meals": total_records,
        "green_alerts": level_summary["SAFE"],
        "yellow_alerts": level_summary["CAUTION"],
        "red_alerts": level_summary["AVOID"],
        "avg_carb_over_ratio": ratio_to_percent(weekly_stats["average_limit_ratios"]["net_carbs"]),
        "avg_gl_over_ratio": ratio_to_percent(weekly_stats["average_limit_ratios"]["gl"]),
        "avg_sodium_over_ratio": ratio_to_percent(weekly_stats["average_limit_ratios"]["sodium"]),
        "avg_salt_over_ratio": ratio_to_percent(weekly_stats["average_limit_ratios"]["salt_equivalent"]),
        "avg_fat_over_ratio": ratio_to_percent(weekly_stats["average_limit_ratios"]["fat"]),
        "avg_sat_fat_over_ratio": ratio_to_percent(weekly_stats["average_limit_ratios"]["saturated_fat"]),
        "avg_cholesterol_over_ratio": ratio_to_percent(weekly_stats["average_limit_ratios"]["cholesterol"]),
        "avg_risk_ratio": weekly_stats["averages"]["risk_ratio_percent"],
        "top_food": sorted_foods[0][0] if sorted_foods else "暂无高频食物",
        "top_risk_reason": sorted_reasons[0][0] if sorted_reasons else "暂无主要风险项"
    }

    return weekly_stats


def estimate_weekly_report(disease_en, summary_data):
    if requests is None:
        error_msg = "requests 未安装，无法生成饮食健康周报文本"
        print(error_msg)
        return False, error_msg

    disease_cn = DISEASE_CN_MAP.get(disease_en, "慢病")
    if not is_ark_text_configured():
        error_msg = "ARK text config missing, weekly report generation unavailable"
        print(error_msg)
        return False, error_msg

    summary_view = {
        "total_meals": summary_data.get("total_meals", 0),
        "green_alerts": summary_data.get("green_alerts", 0),
        "yellow_alerts": summary_data.get("yellow_alerts", 0),
        "red_alerts": summary_data.get("red_alerts", 0),
        "avg_carb_over_ratio_percent": summary_data.get("avg_carb_over_ratio", 0),
        "avg_gl_over_ratio_percent": summary_data.get("avg_gl_over_ratio", 0),
        "avg_sodium_over_ratio_percent": summary_data.get("avg_sodium_over_ratio", 0),
        "avg_fat_over_ratio_percent": summary_data.get("avg_fat_over_ratio", 0),
        "avg_sat_fat_over_ratio_percent": summary_data.get("avg_sat_fat_over_ratio", 0),
        "avg_cholesterol_over_ratio_percent": summary_data.get("avg_cholesterol_over_ratio", 0),
        "avg_risk_ratio_percent": summary_data.get("avg_risk_ratio", 0),
        "top_food": summary_data.get("top_food", "暂无高频食物"),
        "top_risk_reason": summary_data.get("top_risk_reason", "暂无主要风险项")
    }

    system_prompt = (
        f"你是享誉全国的临床营养学主任医师，长期负责{disease_cn}患者的饮食干预与慢病管理。"
        "现在你要根据患者最近一周的膳食统计数据，输出一份 Markdown 格式的【膳食复盘周报】。"
        "报告必须严格包含以下三个一级模块，且模块标题必须逐字一致："
        "【本周膳食红黑榜】、【慢病靶器官风险警示】、【下周定制化膳食处方】。"
        "周报逻辑请严格围绕以下规则展开："
        "SAFE/CAUTION/AVOID 三色警报、最近7天高频食物、主要超标营养素、慢病对应的靶器官风险、以及下周可执行饮食处方。"
        "语气要求权威、专业、具体，不要空泛安慰。"
        "请直接输出最终 Markdown 正文，不要添加任何多余前言、解释或代码块标记。"
    )

    user_prompt = (
        f"请基于以下 {disease_cn} 患者最近 7 天膳食统计数据生成周报：\n"
        f"{json.dumps(summary_view, ensure_ascii=False, indent=2)}\n\n"
        "写作要求：\n"
        "1. 红黑榜要明确指出做得好的地方与最危险的问题。\n"
        "2. 靶器官风险警示要结合该慢病常见受损器官或系统，给出医学化但易懂的说明。\n"
        "3. 下周定制化膳食处方要给出具体、能执行的饮食调整建议。\n"
        "4. 重点围绕净碳水/GL、钠/盐、脂肪/饱和脂肪、胆固醇等慢病关键指标展开。"
    )

    payload = {
        "model": ARK_TEXT_MODEL_ENDPOINT,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.6
    }
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            ARK_CHAT_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=40
        )
        response.raise_for_status()
        response_json = response.json()
        report_text = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if isinstance(report_text, list):
            report_text = "".join(
                item.get("text", "")
                for item in report_text
                if isinstance(item, dict)
            )

        report_text = str(report_text).strip()
        if not report_text:
            error_msg = "周报生成失败：大模型未返回有效内容"
            print(error_msg)
            return False, error_msg

        return True, report_text
    except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError) as exc:
        error_msg = f"周报生成失败: {exc}"
        print(error_msg)
        return False, error_msg


# ==================== 前端页面路由配置 ====================
@app.route('/')
def open_main():
    return render_template("open(3).html")


@app.route('/DCSI')
def dcsi_main():
    return render_template("DCSI(3).html")


@app.route('/base_info')
def base_info():
    return render_template("base_info(2).html")


@app.route('/Diabetes_information_input')
def diabetes_input():
    return render_template("Diabetes_information_input(4).html")


@app.route('/Diabetes_detect_analyse')
def diabetes_analyse():
    return render_template("Diabetes_detect_analyse(3).html")


@app.route('/Hyperglycemia_information_input')
def hyperglycemia_input():
    return render_template("Hyperglycemia_information_input(2).html")


@app.route('/Hyperglycemia_detect_analyse')
def hyperglycemia_analyse():
    return render_template("Hyperglycemia_detect_analyse(2).html")


@app.route('/Hyperlipidemia_information_input')
def hyperlipidemia_input():
    return render_template("Hyperlipidemia_information_input(3).html")


@app.route('/Hyperlipidemia_detect_analyse')
def hyperlipidemia_analyse():
    return render_template("Hyperlipidemia_detect_analyse(3).html")


@app.route('/Hypertension_information_input')
def hypertension_input():
    return render_template("Hypertension_information_input(3).html")


@app.route('/Hypertension_detect_analyse')
def hypertension_analyse():
    return render_template("Hypertension_detect_analyse(3).html")


# ==================== 核心 API：食物识别与分析 ====================
def extract_detection_image_from_request():
    if cv2 is None or np is None:
        raise RuntimeError("当前环境缺少图片解码依赖，无法处理上传图片。")

    image_file = request.files.get("image")
    if image_file is None:
        raise ValueError("缺少 multipart/form-data 字段 image。")

    image_bytes = image_file.read()
    if not image_bytes:
        raise ValueError("上传图片内容为空。")

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("上传图片解码失败，请检查文件格式。")

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("上传图片转换失败。")

    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return frame, img_base64


def capture_detection_frame_from_camera():
    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret:
        raise RuntimeError("摄像头连接失败，请检查硬件连接。")

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("摄像头画面编码失败。")

    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return frame, img_base64


def run_detection_on_frame(frame, img_base64, disease):
    results = model.predict(source=frame, imgsz=640, conf=0.001, verbose=False, device="cpu")
    res = results[0]
    probs = getattr(res, "probs", None)

    if probs is None:
        return empty_detect_response(
            img_base64,
            "未检测到食物",
            "NONE",
            "重新对准菜品",
            "请将摄像头对准菜品，确保菜品在画面中央。"
        )

    best_cls_id = probs.top1
    best_conf = probs.top1conf.item()

    if best_conf < YOLO_CONFIDENCE_THRESHOLD:
        return empty_detect_response(
            img_base64,
            "未检测到食物",
            "NONE",
            "重新对准菜品",
            "请将摄像头对准菜品，确保菜品在画面中央。"
        )

    label = id2name.get(best_cls_id, "未知食物")
    if label in food_db:
        return build_analysis_result(label, food_db[label], disease, img_base64)

    unknown_response = empty_detect_response(
        img_base64,
        label,
        "UNKNOWN",
        "建议少量",
        "暂无该食物的详细饮食建议，请谨慎食用。"
    )
    unknown_response["visual_tip"] = DEFAULT_VISUAL_TIP
    return unknown_response


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    load_food_database()

    if cv2 is None:
        return jsonify(
            empty_detect_response(
                "",
                "摄像头模块缺失",
                "ERROR",
                "检查依赖",
                "当前环境未安装 OpenCV，无法执行识别。"
            )
        )

    try:
        load_model()
    except RuntimeError as exc:
        return jsonify(
            empty_detect_response(
                "",
                "识别模型不可用",
                "ERROR",
                "检查依赖",
                str(exc)
            )
        )

    disease_source = request.form if request.method == "POST" else request.args
    disease = (disease_source.get("disease") or request.args.get("disease") or "diabetes").strip() or "diabetes"
    img_base64 = ""

    try:
        if request.method == "POST":
            frame, img_base64 = extract_detection_image_from_request()
        else:
            frame, img_base64 = capture_detection_frame_from_camera()

        return jsonify(run_detection_on_frame(frame, img_base64, disease))

    except ValueError as exc:
        return jsonify(
            empty_detect_response(
                "",
                "上传图片无效",
                "ERROR",
                "检查上传内容",
                str(exc)
            )
        )
    except RuntimeError as exc:
        if request.method == "POST":
            return jsonify(
                empty_detect_response(
                    img_base64,
                    "图片处理失败",
                    "ERROR",
                    "检查上传内容",
                    str(exc)
                )
            )

        return jsonify(
            empty_detect_response(
                "",
                "摄像头错误",
                "ERROR",
                "检查硬件",
                str(exc)
            )
        )
    except Exception as exc:
        print(f"错误: {exc}")
        return jsonify(
            empty_detect_response(
                img_base64,
                "未检测到食物",
                "NONE",
                "重新对准",
                "识别过程出错，请重试。"
            )
        )


# ==================== 饮食记录 API ====================
@app.route('/api/diet_records', methods=['POST'])
def create_diet_record():
    payload = request.get_json(silent=True) or {}
    user_id = resolve_user_id(payload)
    normalized = normalize_recognition_result(payload)

    if not normalized["food"] or normalized["level"] in {"NONE", "ERROR"}:
        return jsonify({"code": 400, "msg": "缺少有效识别结果，无法记录饮食"}), 400

    disease = normalized["disease"] or "diabetes"
    health_profile = build_request_health_profile(payload, disease)
    recorded_at = payload.get("recorded_at") or now_iso()
    meal_slot = infer_meal_slot(payload.get("meal_slot") or normalized["meal_slot"], recorded_at)
    captured_at = normalized["captured_at"] or recorded_at
    now_text = now_iso()

    nutrition_payload = normalized["nutrition"] or {}
    analysis_payload = dict(normalized["analysis"] or {})

    if not analysis_payload:
        meal_limits = calculate_meal_limits(disease, health_profile, meal_slot)
        ratio_items, risk_ratio = calculate_indicator_ratios(
            nutrition_payload, nutrition_payload, disease, meal_limits
        )
        analysis_payload = {
            "recognition_key": normalized["recognition_key"],
            "captured_at": captured_at,
            "disease": disease,
            "meal_slot": meal_slot,
            "meal_limits": meal_limits,
            "meal_limit_ratios": ratio_items,
            "risk_ratio": risk_ratio,
            "risk_ratio_percent": ratio_to_percent(risk_ratio),
            "main_risk_reason": normalized["main_risk_reason"] or build_main_risk_reason(ratio_items),
            "recommended_grams": normalized["recommendation_grams"],
            "target_ratio": RECOMMENDED_RATIO_TARGET,
            "health_profile": normalize_health_profile(health_profile, disease)
        }

    if not normalized["main_risk_reason"]:
        normalized["main_risk_reason"] = analysis_payload.get("main_risk_reason", "")

    with get_db_connection() as conn:
        existing = conn.execute(
            """
            SELECT *
            FROM diet_records
            WHERE user_id = ? AND recognition_key = ?
            """,
            (user_id, normalized["recognition_key"])
        ).fetchone()

        if existing is None:
            cursor = conn.execute(
                """
                INSERT INTO diet_records (
                    user_id, recognition_key, disease, food_name, status, level, portion_text,
                    tip, visual_tip, actual_weight, percentage, recommended_grams, meal_slot,
                    risk_ratio, main_risk_reason, nutrition_json, analysis_json, health_profile_json,
                    source_payload_json, captured_at, recorded_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    normalized["recognition_key"],
                    disease,
                    normalized["food"],
                    normalized["level"],
                    normalized["portion"],
                    normalized["tip"],
                    normalized["visual_tip"],
                    normalized["actual_weight"],
                    normalized["percentage"],
                    normalized["recommendation_grams"],
                    meal_slot,
                    normalized["risk_ratio"],
                    normalized["main_risk_reason"],
                    json_dumps(nutrition_payload),
                    json_dumps(analysis_payload),
                    json_dumps(health_profile),
                    json_dumps(normalized["source_payload"]),
                    captured_at,
                    recorded_at,
                    now_text,
                    now_text
                )
            )
            record_id = cursor.lastrowid
            row = conn.execute("SELECT * FROM diet_records WHERE id = ?", (record_id,)).fetchone()
            return jsonify({"code": 200, "msg": "success", "data": serialize_record(row)}), 200

        if existing["status"] == "active":
            return jsonify({"code": 200, "msg": "already_recorded", "data": serialize_record(existing)}), 200

        conn.execute(
            """
            UPDATE diet_records
            SET disease = ?, food_name = ?, status = 'active', level = ?, portion_text = ?, tip = ?, visual_tip = ?,
                actual_weight = ?, percentage = ?, recommended_grams = ?, meal_slot = ?, risk_ratio = ?,
                main_risk_reason = ?, nutrition_json = ?, analysis_json = ?, health_profile_json = ?,
                source_payload_json = ?, captured_at = ?, recorded_at = ?, canceled_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (
                disease,
                normalized["food"],
                normalized["level"],
                normalized["portion"],
                normalized["tip"],
                normalized["visual_tip"],
                normalized["actual_weight"],
                normalized["percentage"],
                normalized["recommendation_grams"],
                meal_slot,
                normalized["risk_ratio"],
                normalized["main_risk_reason"],
                json_dumps(nutrition_payload),
                json_dumps(analysis_payload),
                json_dumps(health_profile),
                json_dumps(normalized["source_payload"]),
                captured_at,
                recorded_at,
                now_text,
                existing["id"]
            )
        )
        row = conn.execute("SELECT * FROM diet_records WHERE id = ?", (existing["id"],)).fetchone()
        return jsonify({"code": 200, "msg": "success", "data": serialize_record(row)}), 200


@app.route('/api/diet_records/status', methods=['GET'])
def get_diet_record_status():
    user_id = resolve_user_id()
    recognition_key = request.args.get("recognition_key", "").strip()

    if not recognition_key:
        return jsonify({"code": 400, "msg": "缺少 recognition_key"}), 400

    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM diet_records
            WHERE user_id = ? AND recognition_key = ?
            """,
            (user_id, recognition_key)
        ).fetchone()

    if row is None:
        return jsonify({
            "code": 200,
            "msg": "success",
            "data": {
                "recorded": False,
                "status": "not_recorded",
                "recognition_key": recognition_key
            }
        }), 200

    serialized = serialize_record(row)
    return jsonify({
        "code": 200,
        "msg": "success",
        "data": {
            "recorded": serialized["status"] == "active",
            "status": serialized["status"],
            "recognition_key": recognition_key,
            "record": serialized
        }
    }), 200


@app.route('/api/diet_records/cancel', methods=['POST'])
def cancel_diet_record():
    payload = request.get_json(silent=True) or {}
    user_id = resolve_user_id(payload)
    recognition_key = str(payload.get("recognition_key", "")).strip()
    record_id = safe_int(payload.get("record_id"), 0)

    if not recognition_key and record_id <= 0:
        return jsonify({"code": 400, "msg": "缺少 record_id 或 recognition_key"}), 400

    with get_db_connection() as conn:
        if record_id > 0:
            row = conn.execute(
                "SELECT * FROM diet_records WHERE id = ? AND user_id = ?",
                (record_id, user_id)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM diet_records WHERE user_id = ? AND recognition_key = ?",
                (user_id, recognition_key)
            ).fetchone()

        if row is None:
            return jsonify({"code": 404, "msg": "未找到对应饮食记录"}), 404

        if row["status"] == "canceled":
            return jsonify({"code": 200, "msg": "already_canceled", "data": serialize_record(row)}), 200

        cancel_time = now_iso()
        conn.execute(
            """
            UPDATE diet_records
            SET status = 'canceled', canceled_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (cancel_time, cancel_time, row["id"])
        )
        updated_row = conn.execute("SELECT * FROM diet_records WHERE id = ?", (row["id"],)).fetchone()

    return jsonify({"code": 200, "msg": "success", "data": serialize_record(updated_row)}), 200


@app.route('/api/diet_records/history', methods=['GET'])
def diet_record_history():
    user_id = resolve_user_id()
    disease, include_canceled, page, page_size = get_history_filters()
    total, records = fetch_history_records(user_id, disease, include_canceled, page, page_size)

    return jsonify({
        "code": 200,
        "msg": "success",
        "data": {
            "user_id": user_id,
            "disease": disease,
            "include_canceled": include_canceled,
            "page": page,
            "page_size": page_size,
            "total": total,
            "records": records
        }
    }), 200


# ==================== 周统计与周报 API ====================
@app.route('/api/weekly_stats', methods=['GET'])
def weekly_stats():
    user_id = resolve_user_id()
    disease = request.args.get("disease", "").strip()
    days = min(30, max(1, safe_int(request.args.get("days"), 7)))

    start_dt, end_dt, records = fetch_weekly_records(user_id, disease, days=days, include_canceled=False)
    stats = build_weekly_statistics(records, disease, start_dt, end_dt)

    return jsonify({
        "code": 200,
        "msg": "success",
        "data": {
            "user_id": user_id,
            "records_count": len(records),
            "weekly_stats": stats
        }
    }), 200


@app.route('/api/weekly_report', methods=['POST'])
def weekly_report():
    payload = request.get_json(silent=True) or {}
    user_id = resolve_user_id(payload)
    disease = str(payload.get("disease", "")).strip()
    days = min(30, max(1, safe_int(payload.get("days"), 7)))

    summary_data = payload.get("summary_data")
    weekly_stats = payload.get("weekly_stats")

    if not summary_data:
        start_dt, end_dt, records = fetch_weekly_records(user_id, disease, days=days, include_canceled=False)
        weekly_stats = build_weekly_statistics(records, disease, start_dt, end_dt)
        summary_data = weekly_stats["summary_data"]
    else:
        records = []

    if not summary_data or safe_int(summary_data.get("total_meals"), 0) == 0:
        return jsonify({
            "code": 404,
            "msg": "最近7天暂无可用饮食记录"
        }), 404

    success, result = estimate_weekly_report(disease, summary_data)
    if success:
        return jsonify({
            "code": 200,
            "msg": "success",
            "data": {
                "user_id": user_id,
                "openid": payload.get("openid", ""),
                "weekly_report": result,
                "weekly_stats": weekly_stats,
                "summary_data": summary_data,
                "records_count": len(records) if records is not None else None
            }
        }), 200

    return jsonify({
        "code": 200,
        "msg": "weekly_stats_ready_report_unavailable",
        "data": {
            "user_id": user_id,
            "openid": payload.get("openid", ""),
            "weekly_report": "",
            "report_error": result,
            "weekly_stats": weekly_stats,
            "summary_data": summary_data,
            "records_count": len(records) if records is not None else None
        }
    }), 200


init_database()


# ==================== 程序主入口 ====================
if __name__ == "__main__":
    load_food_database()
    try:
        load_model()
    except RuntimeError as exc:
        print(f"\n⚠️ {exc}")
    print(f"\n✅ SQLite 数据库就绪：{DB_PATH}")
    print("\n✅ 系统启动成功！访问页面即可使用")
    app.run(host="0.0.0.0", port=5000, debug=False)
