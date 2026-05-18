# ==============================================
# NutriLens 智能饮食识别与慢病管理系统
# 核心后端程序 (Flask + YOLOv8 + 营养数据库)
# 功能：实现食物AI识别、分病种营养分析、饮食建议推送
# ==============================================

# 1. 导入项目所需的所有第三方库
import cv2               # 开源计算机视觉库，用于摄像头画面读取、图像处理
import json              # JSON数据解析库，用于加载食物营养数据库
import base64            # Base64编码库，用于将图片转为字符串传输给前端
import os                # 系统环境配置库，用于禁用GPU

# 禁用CUDA GPU加速，强制使用CPU运行（适配Jetson/普通电脑）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from ultralytics import YOLO    # YOLOv8目标检测框架，用于食物图像识别
from flask import Flask, jsonify, render_template, request  # Flask Web框架核心组件

# 2. 初始化Flask Web应用实例
app = Flask(__name__)

# ==================== 全局配置数据 ====================
# 默认健康参数（前端未提交用户数据时，使用该默认值计算饮食建议）
default_health_params = {
    "daily_net_carbs": 80.0,    # 每日净碳水化合物推荐摄入量(g)
    "fasting_sugar": 6.5,       # 空腹血糖默认参考值(mmol/L)
    "daily_fat": 50.0,          # 每日脂肪推荐摄入量(g)
    "cholesterol": 5.2,         # 胆固醇默认参考值(mmol/L)
    "daily_sodium": 2000.0,     # 每日钠推荐摄入量(mg)
    "blood_pressure": 140       # 血压默认参考值(mmHg)
}

food_db = {}  # 全局食物营养数据库字典，存储所有菜品的营养数据

# 菜品ID与名称映射表（YOLO模型输出ID → 对应食物中文名称）
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

model = None  # 全局YOLO模型变量，后续加载训练好的权重文件

# ==================== 核心算法：饮食建议生成函数 ====================
# 功能：根据食物营养数据 + 用户健康参数，计算食用等级和推荐食用量
# 参数：food_nutri - 单种食物的完整营养数据字典
# 返回：level(食用等级)、portion(推荐食用量)
def get_food_suggestion(food_nutri):
    # 提取食物核心营养指标
    net_carbs = food_nutri["net_carbs"]
    gl = food_nutri["gl"]
    # 提取默认健康参数
    daily_carbs = default_health_params["daily_net_carbs"]
    sugar = default_health_params["fasting_sugar"]

    # 无碳水化合物：无限制食用
    if net_carbs <= 0:
        portion = "无限制食用"
    else:
        # 计算最大推荐食用克数
        max_gram = (daily_carbs * 0.2 / net_carbs) * 100
        portion = f"最大推荐 {int(max_gram)} 克"

    # 根据血糖负荷(GL)和净碳水，判定食用安全等级
    if gl < 10 and net_carbs < 15:
        level = "SAFE"      # 安全
    elif gl < 20 and net_carbs < 30:
        level = "CAUTION"   # 谨慎
    else:
        level = "AVOID"     # 避免

    # 血糖偏高时，减半食用量
    if sugar > 7.0:
        portion = f"血糖偏高，建议减半食用" if net_carbs > 0 else "严格控制摄入"
    return level, portion

# ==================== 前端页面路由配置 ====================
# 根路由：项目启动首页(open.html)
@app.route('/')
def open_main():
    return render_template("open.html")

# 主菜单页面路由
@app.route('/DCSI')
def dcsi_main():
    return render_template("DCSI.html")

# 糖尿病模块路由
@app.route('/Diabetes_information_input')
def diabetes_input():
    return render_template("Diabetes_information_input.html")
@app.route('/Diabetes_detect_analyse')
def diabetes_analyse():
    return render_template("Diabetes_detect_analyse.html")

# 高血糖模块路由
@app.route('/Hyperglycemia_information_input')
def hyperglycemia_input():
    return render_template("Hyperglycemia_information_input.html")
@app.route('/Hyperglycemia_detect_analyse')
def hyperglycemia_analyse():
    return render_template("Hyperglycemia_detect_analyse.html")

# 高血脂模块路由
@app.route('/Hyperlipidemia_information_input')
def hyperlipidemia_input():
    return render_template("Hyperlipidemia_information_input.html")
@app.route('/Hyperlipidemia_detect_analyse')
def hyperlipidemia_analyse():
    return render_template("Hyperlipidemia_detect_analyse.html")

# 高血压模块路由
@app.route('/Hypertension_information_input')
def hypertension_input():
    return render_template("Hypertension_information_input.html")
@app.route('/Hypertension_detect_analyse')
def hypertension_analyse():
    return render_template("Hypertension_detect_analyse.html")

# ==================== 核心API接口：食物识别与分析 ====================
# 接口地址：/detect
# 功能：调用摄像头采集画面 → YOLO识别食物 → 匹配营养数据 → 返回前端结果
@app.route('/detect')
def detect():
    # 获取前端传递的病种参数，默认糖尿病
    disease = request.args.get("disease", "diabetes")

    # 初始化摄像头，设置分辨率
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # 读取一帧画面
    ret, frame = cap.read()
    # 释放摄像头资源
    cap.release()

    # 摄像头读取失败：返回错误信息
    if not ret:
        return jsonify({
            "imgBase64": "", "food": "摄像头错误", "level": "ERROR", "portion": "检查硬件",
            "cal": "---", "net_carbs": "---", "gl": "---", "gi": "---",
            "fat": "---", "total_fat": "---",
            "saturated_fat": "---", "cholesterol": "---",
            "sodium": "---", "salt_equivalent": "---", "saltEquivalent": "---", "salt": "---",
            "tip": "摄像头连接失败，请检查硬件连接。"
        })

    # 将画面转为JPG格式，再编码为Base64字符串
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    try:
        # 调用YOLO模型进行食物识别
        results = model.predict(source=frame, imgsz=640, conf=0.001, verbose=False, device="cpu")
        res = results[0]
        # 获取置信度最高的食物ID和置信度
        best_cls_id = res.probs.top1
        best_conf = res.probs.top1conf.item()

        # 置信度过低：未检测到有效食物
        if best_conf < 0.25:
            return jsonify({
                "imgBase64": img_base64, "food": "未检测到食物", "level": "NONE", "portion": "重新对准菜品",
                "cal": "---", "net_carbs": "---", "gl": "---", "gi": "---",
                "fat": "---", "total_fat": "---",
                "saturated_fat": "---", "cholesterol": "---",
                "sodium": "---", "salt_equivalent": "---", "saltEquivalent": "---", "salt": "---",
                "tip": "请将摄像头对准菜品，确保菜品在画面中央。"
            })

        # 根据ID获取食物名称
        label = id2name.get(best_cls_id, "未知食物")
        # 匹配营养数据库，存在则生成详细建议
        if label in food_db:
            nutri = food_db[label]
            # 调用算法，获取食用等级和推荐量
            level, portion = get_food_suggestion(nutri)

            # 根据病种，匹配对应的饮食建议(tip1/2/3)
            if disease in ["diabetes", "hyperglycemia"]:
                tip = nutri["tip1"]
            elif disease == "hyperlipidemia":
                tip = nutri["tip2"]
            elif disease == "hypertension":
                tip = nutri["tip3"]
            else:
                tip = nutri["tip1"]

            # 兼容前端所有字段名，统一赋值
            fat_val = str(nutri.get("fat", "---"))
            salt_val = str(nutri.get("salt_equivalent", "---"))

            # 返回完整的识别结果给前端
            return jsonify({
                "imgBase64": img_base64,
                "food": label,
                "level": level,
                "portion": portion,
                "cal": str(nutri.get("calorie", "---")),
                "net_carbs": str(nutri.get("net_carbs", "---")),
                "gl": str(nutri.get("gl", "---")),
                "gi": str(nutri.get("gi", "---")),
                "fat": fat_val,
                "total_fat": fat_val,
                "saturated_fat": str(nutri.get("saturated_fat", "---")),
                "cholesterol": str(nutri.get("cholesterol", "---")),
                "sodium": str(nutri.get("sodium", "---")),
                "salt_equivalent": salt_val,
                "saltEquivalent": salt_val,
                "salt": salt_val,
                "tip": tip
            })
        else:
            # 无营养数据：返回未知食物提示
            return jsonify({
                "imgBase64": img_base64, "food": label, "level": "UNKNOWN", "portion": "建议少量",
                "cal": "---", "net_carbs": "---", "gl": "---", "gi": "---",
                "fat": "---", "total_fat": "---",
                "saturated_fat": "---", "cholesterol": "---",
                "sodium": "---", "salt_equivalent": "---", "saltEquivalent": "---", "salt": "---",
                "tip": "暂无该食物的详细饮食建议，请谨慎食用。"
            })

    except Exception as e:
        # 程序异常捕获：打印错误并返回提示
        print(f"错误: {str(e)}")
        return jsonify({
            "imgBase64": img_base64, "food": "未检测到食物", "level": "NONE", "portion": "重新对准",
            "cal": "---", "net_carbs": "---", "gl": "---", "gi": "---",
            "fat": "---", "total_fat": "---",
            "saturated_fat": "---", "cholesterol": "---",
            "sodium": "---", "salt_equivalent": "---", "saltEquivalent": "---", "salt": "---",
            "tip": "识别过程出错，请重试。"
        })

# ==================== 程序主入口 ====================
if __name__ == "__main__":
    print("\n�� 加载营养数据库...")
    food_db = {}

    import json
    from json.decoder import JSONDecoder

    # 读取本地食物营养JSON文件
    with open("food_nutrition.json", "r", encoding="utf-8") as f:
        content = f.read()

    decoder = JSONDecoder()
    pos = 0
    content_len = len(content)

    # 容错解析JSON数据，兼容格式异常
    while pos < content_len:
        try:
            obj, new_pos = decoder.raw_decode(content, pos)
            if isinstance(obj, dict) and len(obj) > 0:
                food_db.update(obj)
            pos = new_pos
        except json.JSONDecodeError:
            pos += 1

    print(f"✅ 营养数据库加载完成，共 {len(food_db)} 种食物")

    print("\n�� 加载YOLO模型...")
    # 加载训练好的YOLO模型权重文件
    model = YOLO("model/best.pt")
    print("✅ 模型加载成功！")

    print("\n✅ 系统启动成功！访问页面即可使用")
    # 启动Flask服务，允许局域网所有设备访问
    app.run(host="0.0.0.0", port=5000, debug=False)
