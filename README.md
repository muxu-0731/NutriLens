# NutriLens

NutriLens is a Flask-based food recognition and chronic-disease dietary analysis system built for mobile web scenarios.

It is designed around a real scanning workflow:

**WeChat / H5 page -> mobile camera capture -> image upload -> YOLO recognition -> ARK portion analysis -> nutrition and risk feedback**

The current version focuses on four chronic disease scenarios:

- Diabetes
- Hyperglycemia
- Hyperlipidemia
- Hypertension

## Highlights

- Real mobile photo upload via `POST /detect`
- YOLOv8 food recognition
- ARK-based visual portion estimation
- Disease-specific nutrition and risk analysis
- Diet record persistence with SQLite
- Weekly stats and weekly report generation
- Mobile-first analysis pages with real scan flow

## Current Workflow

1. User opens a disease-specific analysis page on a mobile browser.
2. User taps the scan button and takes a photo with the phone camera.
3. Frontend uploads the image to `POST /detect` using `multipart/form-data`.
4. Backend runs:
   - image decoding
   - YOLO recognition
   - food label mapping
   - ARK portion estimation
   - nutrition calculation
   - disease-specific risk analysis
5. Frontend renders the recognition result and supports rescan.

## Core Capabilities

### 1. Real food detection

- `POST /detect` accepts uploaded images from browsers
- `GET /detect` is kept as a compatibility/debug mode using `cv2.VideoCapture(0)`
- Shared frontend scan flow in `static/detect_realtime.js`

### 2. Nutrition and portion analysis

- Food recognition based on YOLOv8
- Nutrition mapping from `food_nutrition.json`
- Portion percentage estimated from the uploaded image through ARK
- Actual grams derived from portion ratio and reference serving size

### 3. Disease-oriented analysis

- Outputs risk levels such as `SAFE`, `CAUTION`, and `AVOID`
- Returns recommendation grams, risk ratio, main risk reason, and visual hints
- Preserves the existing nutrition analysis and risk analysis pipeline

### 4. Record and report features

- Create diet records
- Query record status
- Cancel diet records
- View history
- Generate weekly statistics
- Generate weekly text reports

## Tech Stack

- Python 3.8+
- Flask
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Requests
- SQLite
- ARK vision/text APIs

## Project Structure

```text
NutriLens-pro/
|- food_detector.py                  # Main backend application
|- food_nutrition.json               # Local food nutrition database
|- model/
|  `- best.pt                        # YOLO model weights
|- data/
|  `- nutrilens.db                   # SQLite database
|- static/
|  |- detect_realtime.js             # Shared realtime mobile scan script
|  `- logo.png
|- templates/
|  |- open(3).html
|  |- DCSI(3).html
|  |- base_info(2).html
|  |- Diabetes_information_input(4).html
|  |- Diabetes_detect_analyse(3).html
|  |- Hyperglycemia_information_input(2).html
|  |- Hyperglycemia_detect_analyse(2).html
|  |- Hyperlipidemia_information_input(3).html
|  |- Hyperlipidemia_detect_analyse(3).html
|  |- Hypertension_information_input(3).html
|  `- Hypertension_detect_analyse(3).html
`- README.md
```

## Requirements

- Python 3.8 or later
- A valid YOLO model file at `model/best.pt`
- A valid food nutrition database file at `food_nutrition.json`
- Network access for ARK-dependent features such as portion estimation and weekly report generation

## Installation

Install dependencies:

```bash
pip install flask opencv-python numpy requests ultralytics
```

## Configuration

NutriLens now loads configuration in this order:

1. System environment variables on the server
2. Local project `.env` file for development

For local development:

1. Copy `.env.example`
2. Rename the copied file to `.env`
3. Fill in your own ARK key and endpoint values

The real `.env` file is ignored by Git and should stay local only.

Recommended environment variables:

```bash
ARK_API_KEY=your_api_key
ARK_VISION_URL=https://ark.cn-beijing.volces.com/api/v3/responses
ARK_VISION_MODEL_ENDPOINT=your_vision_endpoint
ARK_CHAT_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_TEXT_MODEL_ENDPOINT=your_text_endpoint
```

Optional:

```bash
CUDA_VISIBLE_DEVICES=-1
```

The current backend forces CPU mode by default for broader compatibility.

If ARK vision variables are missing, food detection can still run, but portion estimation falls back to a default value.

If ARK text variables are missing, weekly report generation will be unavailable.

## Run

```bash
python food_detector.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Main Pages

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

## API Overview

### `POST /detect`

Primary production mode for real mobile scanning.

Content type:

```text
multipart/form-data
```

Fields:

- `image`: uploaded image file
- `disease`: `diabetes` / `hyperglycemia` / `hyperlipidemia` / `hypertension`

Example:

```bash
curl -X POST "http://127.0.0.1:5000/detect" \
  -F "image=@test.jpg" \
  -F "disease=diabetes"
```

### `GET /detect`

Compatibility/debug mode that captures from the server camera with `cv2.VideoCapture(0)`.

Example:

```bash
curl "http://127.0.0.1:5000/detect?disease=diabetes"
```

### `POST /api/diet_records`

Create or restore a diet record using recognition data.

### `GET /api/diet_records/status`

Check whether a recognition result has already been recorded.

### `POST /api/diet_records/cancel`

Cancel an existing diet record.

### `GET /api/diet_records/history`

Query paginated history records.

### `GET /api/weekly_stats`

Generate weekly nutrition and risk statistics.

### `POST /api/weekly_report`

Generate a weekly summary report.

## Detection Response Fields

The frontend consumes fields such as:

- `food`
- `level`
- `portion`
- `tip`
- `visual_tip`
- `recommendation_grams`
- `risk_ratio`
- `main_risk_reason`
- `nutrition`
- `analysis`
- `imgBase64`

## Algorithm Upgrades in the Current Version

Compared with the earliest prototype, the current system has moved beyond simple label-based suggestion logic and now includes:

- real uploaded-image recognition
- portion percentage estimation
- actual gram estimation
- scalable nutrition calculation
- disease-specific meal-limit logic
- risk-ratio-based warning evaluation
- recommendation gram back-calculation
- weekly aggregation and report generation

## Deployment Notes

- `POST /detect` is the recommended mode for real deployment.
- `GET /detect` should be treated as local compatibility/debug mode only.
- SQLite tables are initialized automatically at startup.
- If ARK services are unavailable, portion estimation or weekly report generation may fail gracefully.

## Recommended Use Cases

- WeChat Official Account H5 pages
- Mobile diet analysis tools
- Chronic disease dietary education demos
- Food recognition and nutrition analysis prototypes

## Roadmap

- Better multi-disease joint evaluation
- Stronger health-profile participation in realtime analysis
- Improved internationalization and accessibility
- OpenAPI documentation
- Dockerized deployment

## Publishing Advice

Before pushing this project to GitHub:

- keep real secrets only in server environment variables or local `.env`
- confirm whether model weights and database files should be committed
- add a proper open-source license if you plan to publish publicly

## License

Add your preferred license before public release.
