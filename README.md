# NutriLens

## AI智慧营养视界 / 智能饮食识别与慢病管理系统

NutriLens 是一个面向移动健康管理场景构建的 AI 智能饮食识别与慢病管理系统，聚焦糖尿病、高血糖、高血脂、高血压等慢性病相关饮食辅助分析需求。项目以食物图像识别、营养映射分析、慢病风险评估和饮食行为追踪为核心，形成了从前端采集、云端识别、个性化分析到结果反馈与记录管理的完整业务闭环。

本项目已完成从 **PC 端单次识别原型** 向 **手机拍照上传、云端识别、慢病个性化分析、安全克数推荐、饮食记录与周报追踪** 的完整升级，能够更贴近真实移动端健康管理场景，适合作为智能营养分析、慢病饮食辅助、健康管理产品原型与教学展示项目使用。

## 项目简介

NutriLens 采用 Flask 构建后端服务，结合 YOLOv8 食物识别能力、本地营养数据库以及 ARK 多模态/文本能力，对用户上传的膳食图片进行识别、分量估算、营养分析与慢病适配性判断。系统支持手机端拍照上传图片，后端完成识别与分析后，向前端返回风险等级、推荐摄入克数、主要风险原因、营养明细及可视化提示，并支持饮食记录、历史追踪、周统计与周报生成。

项目当前重点覆盖以下慢病饮食管理场景：

- 糖尿病饮食分析
- 高血糖饮食分析
- 高血脂饮食分析
- 高血压饮食分析

## 核心功能

- 支持手机端拍照或上传图片，完成真实饮食场景下的食物识别
- 基于 YOLOv8 对食物目标进行识别与标签映射
- 结合 ARK 视觉能力完成餐食分量比例估算
- 基于 `food_nutrition.json` 进行营养成分换算与克数推算
- 面向不同慢病场景输出个性化风险等级与饮食建议
- 返回推荐摄入克数、主要风险因素、风险比值与可视化提示信息
- 支持饮食记录创建、状态查询、取消、历史查看等记录管理功能
- 支持每周饮食统计与周报生成，满足持续追踪需求

## 技术亮点

- 完成了从本地摄像头调试模式到移动端真实上传识别流程的升级
- 实现了“食物识别 + 分量估算 + 营养计算 + 慢病分析 + 记录追踪”的一体化闭环
- 兼顾实时识别体验与慢病个性化分析逻辑，不仅识别“吃了什么”，还分析“是否适合吃、建议吃多少”
- 支持按疾病场景输出差异化结果，增强健康建议的针对性
- 引入周统计与周报能力，使系统从单次识别工具扩展为连续饮食管理辅助系统
- 后端支持在缺少部分 ARK 能力时进行降级处理，提升项目演示与部署兼容性

## 技术栈

- Python 3.8+
- Flask
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Requests
- SQLite
- HTML / CSS / JavaScript
- ARK Vision / Text API

## 项目目录结构

```text
NutriLens-pro/
├─ food_detector.py                         # Flask 后端主程序
├─ food_nutrition.json                      # 食物营养数据库
├─ .env.example                             # 环境变量示例文件
├─ model/
│  └─ best.pt                               # YOLO 模型权重文件
├─ static/
│  ├─ detect_realtime.js                    # 移动端拍照上传与识别交互脚本
│  └─ logo.png                              # 项目图形资源
├─ templates/
│  ├─ open(3).html
│  ├─ DCSI(3).html
│  ├─ base_info(2).html
│  ├─ Diabetes_information_input(4).html
│  ├─ Diabetes_detect_analyse(3).html
│  ├─ Hyperglycemia_information_input(2).html
│  ├─ Hyperglycemia_detect_analyse(2).html
│  ├─ Hyperlipidemia_information_input(3).html
│  ├─ Hyperlipidemia_detect_analyse(3).html
│  ├─ Hypertension_information_input(3).html
│  └─ Hypertension_detect_analyse(3).html
├─ program_mermaid/                         # 流程图与说明素材
├─ alg_update.docx                          # 算法更新说明文档
└─ README.md
```

说明：

- `data/` 目录为运行过程中自动创建的本地数据目录，通常用于保存 SQLite 数据文件。
- `.env` 为本地开发配置文件，不应提交到公开仓库。

## 环境变量配置说明

如需本地体验，请先复制 `.env.example` 为 `.env`，再根据自身环境填写所需配置项。请勿将真实密钥提交到 GitHub。

推荐步骤如下：

```bash
cp .env.example .env
```

Windows PowerShell 可使用：

```powershell
Copy-Item .env.example .env
```

请在 `.env` 中自主填写以下内容：

```bash
ARK_API_KEY=your_ark_api_key
ARK_VISION_URL=https://ark.cn-beijing.volces.com/api/v3/responses
ARK_VISION_MODEL_ENDPOINT=your_vision_model_endpoint
ARK_CHAT_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_TEXT_MODEL_ENDPOINT=your_text_model_endpoint
```

说明：

- `ARK_API_KEY`：ARK 服务访问密钥
- `ARK_VISION_MODEL_ENDPOINT`：视觉分析所需模型 Endpoint
- `ARK_TEXT_MODEL_ENDPOINT`：文本报告生成所需模型 Endpoint
- `ARK_VISION_URL` 与 `ARK_CHAT_URL`：分别对应视觉与文本能力请求地址

项目默认使用 CPU 运行，以兼容更多本地演示环境。如需额外控制设备，可结合环境变量自行调整。

## 本地运行方式

### 1. 安装依赖

```bash
pip install flask opencv-python numpy requests ultralytics
```

### 2. 准备模型与配置

- 确认 `model/best.pt` 已存在
- 确认 `food_nutrition.json` 已存在
- 按上一节完成 `.env` 配置

### 3. 启动项目

```bash
python food_detector.py
```

默认启动后可访问：

```text
http://127.0.0.1:5000
```

主要页面入口包括：

- `/`
- `/DCSI`
- `/base_info`
- `/Diabetes_information_input`
- `/Diabetes_detect_analyse`
- `/Hyperglycemia_information_input`
- `/Hyperglycemia_detect_analyse`
- `/Hyperlipidemia_information_input`
- `/Hyperlipidemia_detect_analyse`
- `/Hypertension_information_input`
- `/Hypertension_detect_analyse`

## 在线体验地址

[http://118.89.77.210:5000]

## Release 版本说明

### 当前版本特点

- 已完成从 PC 端单次识别演示到移动端真实拍照上传识别流程的升级
- 已接入云端分量估算与慢病个性化分析能力
- 已支持安全克数推荐、饮食记录管理、每周统计与周报追踪

### 版本演进概述

- `Prototype`：以本地识别与基础建议为主的早期原型
- `Mobile Upgrade`：完成移动端上传识别与统一前端交互流程
- `Management Upgrade`：引入慢病分析、推荐克数、饮食记录与周报功能

如后续发布正式 Release，可在 GitHub Releases 中补充版本号、变更日志与部署说明。

## 注意事项与免责声明

### 注意事项

- 本项目为研究、教学、展示与原型验证用途，使用前请确认本地模型、依赖与环境变量配置完整
- `.env`、数据库文件及其他本地运行产物不应提交到公开仓库
- 若未正确配置 ARK 相关能力，部分分量估算或周报生成功能可能不可用或降级运行
- 模型识别结果会受到拍摄角度、光照条件、图像清晰度及样本覆盖范围影响

### 免责声明

本项目提供的是饮食健康管理辅助建议，仅用于健康教育、饮食参考与系统演示，不替代医生诊断、治疗方案或注册营养师提供的个体化处方。用户在面对疾病诊疗、药物调整、特殊人群营养干预等问题时，应以专业医疗机构和持证专业人员意见为准。
