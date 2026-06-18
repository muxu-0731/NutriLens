(function () {
    const config = Object.assign({
        detectEndpoint: "/detect",
        diseaseKey: "diabetes",
        inputPage: "/",
        introSpeech: "",
        backSpeech: "返回输入页。",
        scanPrompt: "点击下方按钮开始拍照识别",
        staticTips: []
    }, window.PAGE_CONFIG || {});

    const viewBox = document.getElementById("view-box");
    const scanFocus = viewBox ? viewBox.querySelector(".scan-focus") : null;
    const statusText = document.getElementById("statusText");
    const riskBadge = document.getElementById("riskBadge");
    const conclusionText = document.getElementById("conclusionText");
    const reasonText = document.getElementById("reasonText");
    const recommendedGrams = document.getElementById("recommendedGrams");
    const portionHint = document.getElementById("portionHint");
    const plateWeight = document.getElementById("plateWeight");
    const foodName = document.getElementById("foodName");
    const riskNutrient = document.getElementById("riskNutrient");
    const diseaseReminder = document.getElementById("diseaseReminder");
    const priorityList = document.getElementById("priorityList");
    const detailGrid = document.getElementById("detailGrid");
    const detailPanel = document.getElementById("detailPanel");
    const detailToggle = document.getElementById("detailToggle");
    const tipList = document.getElementById("tipList");
    const mealMode = document.getElementById("mealMode");
    const rescanBtn = document.getElementById("rescanBtn");
    const pageTitle = document.querySelector("h1");

    if (!viewBox || !scanFocus || !statusText || !riskBadge || !conclusionText || !reasonText || !recommendedGrams
        || !portionHint || !plateWeight || !foodName || !riskNutrient || !diseaseReminder || !priorityList
        || !detailGrid || !detailPanel || !detailToggle || !tipList || !mealMode || !rescanBtn) {
        return;
    }

    const defaultScanMarkup = scanFocus.innerHTML;
    const defaultTitle = conclusionText.textContent;
    const defaultReason = reasonText.textContent;
    const defaultMealMode = mealMode.textContent;
    const defaultStatus = config.scanPrompt;
    const defaultPageTitle = pageTitle ? pageTitle.textContent.trim() : "识别分析";

    let latestResult = null;
    let hasScanned = false;
    let lastPreviewUrl = "";
    let activeObjectUrl = "";

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.setAttribute("capture", "environment");
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    function pickChineseVoice() {
        const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
        return voices.find((voice) => /zh|chinese|xiaoxiao|xiaoyi|tingting|female/i.test((voice.lang || "") + " " + (voice.name || "")))
            || voices.find((voice) => /zh/i.test(voice.lang || "")) || null;
    }

    function autoSpeak(text) {
        if (!("speechSynthesis" in window) || !text) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "zh-CN";
        utterance.rate = 0.9;
        const voice = pickChineseVoice();
        if (voice) utterance.voice = voice;
        window.speechSynthesis.speak(utterance);
    }

    function formatNumber(value, digits) {
        const num = Number(value);
        if (!Number.isFinite(num)) return "--";
        return num.toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*[1-9])0+$/, "$1");
    }

    function formatWithUnit(value, unit, digits) {
        const text = formatNumber(value, digits);
        return text === "--" ? `-- ${unit}` : `${text} ${unit}`;
    }

    function levelTheme(level) {
        const key = String(level || "").toUpperCase();
        if (key === "SAFE") return "safe";
        if (key === "CAUTION" || key === "NONE" || key === "UNKNOWN") return "caution";
        return "avoid";
    }

    function readableLevel(level) {
        const key = String(level || "").toUpperCase();
        if (key === "SAFE") return "推荐食用";
        if (key === "CAUTION") return "建议适量食用";
        if (key === "AVOID") return "建议减少或避免本次分量";
        if (key === "UNKNOWN") return "识别到食物，但暂无完整营养建议";
        if (key === "NONE") return "未检测到有效食物";
        if (key === "ERROR") return "本次识别失败";
        return "等待识别结果";
    }

    function mealSlotLabel(slot) {
        const labels = {
            breakfast: "早餐场景",
            lunch: "午餐场景",
            dinner: "晚餐场景",
            snack: "加餐场景"
        };
        return labels[String(slot || "").toLowerCase()] || "真实拍照识别";
    }

    function clearObjectPreview() {
        if (activeObjectUrl) {
            URL.revokeObjectURL(activeObjectUrl);
            activeObjectUrl = "";
        }
    }

    function setPreviewMarkup(markup) {
        scanFocus.innerHTML = markup;
    }

    function setPreviewSource(src, alt) {
        if (!src) {
            lastPreviewUrl = "";
            clearObjectPreview();
            setPreviewMarkup(defaultScanMarkup);
            return;
        }
        lastPreviewUrl = src;
        setPreviewMarkup(`<img src="${src}" alt="${alt || "识别图片"}" style="width:100%;height:100%;object-fit:cover;border-radius:18px;display:block;">`);
    }

    function setLocalPreview(file) {
        clearObjectPreview();
        activeObjectUrl = URL.createObjectURL(file);
        setPreviewSource(activeObjectUrl, "待识别图片");
    }

    function setRemotePreview(imgBase64) {
        clearObjectPreview();
        if (!imgBase64) return;
        setPreviewSource(`data:image/jpeg;base64,${imgBase64}`, "识别结果图片");
    }

    function buildSpeakText(payload) {
        const food = payload.food || "当前餐食";
        const levelText = readableLevel(payload.level);
        const grams = Number(payload.recommendation_grams || payload.analysis?.recommended_grams || 0);
        const weight = Number(payload.actual_weight || payload.nutrition?.actual_weight || 0);
        const reason = payload.main_risk_reason ? `主要关注${payload.main_risk_reason}。` : "";
        const tip = payload.tip ? `${payload.tip}` : "";
        const gramText = grams > 0 ? `推荐食用 ${formatNumber(grams, 0)} 克。` : "";
        const weightText = weight > 0 ? `当前估算分量约 ${formatNumber(weight, 0)} 克。` : "";
        return `${food}，${levelText}。${gramText}${weightText}${reason}${tip}`;
    }

    function setBusy(isBusy, message) {
        viewBox.classList.toggle("scanning", Boolean(isBusy));
        rescanBtn.disabled = Boolean(isBusy);
        rescanBtn.textContent = isBusy ? "识别中..." : (hasScanned ? "重新扫描" : "开始扫描");
        statusText.textContent = message;
    }

    function setBadge(level) {
        const text = String(level || "SAFE").toUpperCase();
        riskBadge.className = `badge ${levelTheme(text)}`;
        riskBadge.textContent = text;
    }

    function renderPlaceholderPriority(message) {
        priorityList.innerHTML = `
            <div class="priority-item">
                <div class="priority-top">
                    <strong>等待识别结果</strong>
                    <span>--</span>
                </div>
                <div class="priority-sub">${message}</div>
                <div class="progress-track"><div class="progress-fill safe" style="width:0%"></div></div>
            </div>
        `;
    }

    function renderPriority(payload) {
        const items = Array.isArray(payload.analysis && payload.analysis.meal_limit_ratios)
            ? payload.analysis.meal_limit_ratios : [];

        if (!items.length) {
            renderPlaceholderPriority("识别完成后将在这里显示本餐优先关注的营养项。");
            return;
        }

        priorityList.innerHTML = items.map((item) => {
            const ratio = Number(item.ratio || 0);
            const percent = Math.max(0, Math.min(100, Number(item.percent || ratio * 100 || 0)));
            return `
                <div class="priority-item">
                    <div class="priority-top">
                        <strong>${item.label || "风险项"}</strong>
                        <span>${formatNumber(item.actual_value, 1)} / ${formatNumber(item.limit_value, 1)}</span>
                    </div>
                    <div class="priority-sub">占本餐上限 ${formatNumber(percent, 0)}%</div>
                    <div class="progress-track">
                        <div class="progress-fill ${levelTheme(ratio <= 0.3 ? "SAFE" : ratio <= 0.7 ? "CAUTION" : "AVOID")}" style="width:${percent}%"></div>
                    </div>
                </div>
            `;
        }).join("");
    }

    function renderDetails(payload) {
        const nutrition = payload.nutrition || {};
        const analysis = payload.analysis || {};
        const detailItems = [
            { label: "估算重量", value: formatWithUnit(nutrition.actual_weight, "g", 0) },
            { label: "分量占比", value: nutrition.percentage ? `${formatNumber(nutrition.percentage, 0)}%` : "--" },
            { label: "热量", value: formatWithUnit(nutrition.calorie, "kcal", 0) },
            { label: "净碳水", value: formatWithUnit(nutrition.net_carbs, "g", 1) },
            { label: "GL", value: formatNumber(nutrition.gl, 1) },
            { label: "GI", value: formatNumber(nutrition.gi, 0) },
            { label: "脂肪", value: formatWithUnit(nutrition.fat, "g", 1) },
            { label: "饱和脂肪", value: formatWithUnit(nutrition.saturated_fat, "g", 1) },
            { label: "胆固醇", value: formatWithUnit(nutrition.cholesterol, "mg", 0) },
            { label: "钠", value: formatWithUnit(nutrition.sodium, "mg", 0) },
            { label: "盐当量", value: formatWithUnit(nutrition.salt_equivalent, "g", 1) },
            { label: "风险比例", value: analysis.risk_ratio_percent ? `${formatNumber(analysis.risk_ratio_percent, 0)}%` : "--" }
        ];

        detailGrid.innerHTML = detailItems.map((item) => `
            <div>
                <strong>${item.label}</strong>
                <span>${item.value}</span>
            </div>
        `).join("");
    }

    function renderTips(payload) {
        const tips = [];
        if (payload.visual_tip) tips.push(`分量参考：${payload.visual_tip}`);
        if (payload.tip) tips.push(payload.tip);
        if (payload.main_risk_reason) tips.push(`本次主要风险关注：${payload.main_risk_reason}`);

        if (String(payload.level || "").toUpperCase() === "UNKNOWN") {
            tips.push("当前食物已识别，但暂无完整营养库映射，请结合分量谨慎参考。");
        }
        if (String(payload.level || "").toUpperCase() === "NONE") {
            tips.push("请将菜品放在画面中央，并确保光线充足后重新扫描。");
        }
        if (String(payload.level || "").toUpperCase() === "ERROR") {
            tips.push("识别过程中发生异常，请检查网络或稍后重试。");
        }

        (config.staticTips || []).forEach((tip) => {
            if (tip) tips.push(tip);
        });

        const uniqueTips = [...new Set(tips.filter(Boolean))];
        tipList.innerHTML = uniqueTips.map((tip) => `<li>${tip}</li>`).join("");
    }

    function getConclusionTitle(payload) {
        const food = payload.food || "本次识别";
        const levelText = readableLevel(payload.level);
        return `${food} · ${levelText}`;
    }

    function getReasonText(payload) {
        const level = String(payload.level || "").toUpperCase();
        if (payload.tip) return payload.tip;
        if (level === "NONE") return "当前没有识别到明确食物，请重新拍摄更清晰的餐盘画面。";
        if (level === "ERROR") return "上传或识别过程发生异常，请稍后重新扫描。";
        if (level === "UNKNOWN") return "当前食物未命中完整营养数据库，请结合分量建议谨慎判断。";
        return payload.visual_tip || defaultReason;
    }

    function getStatusText(payload) {
        const level = String(payload.level || "").toUpperCase();
        if (level === "SAFE" || level === "CAUTION" || level === "AVOID") {
            return "识别完成，可下滑查看完整营养与风险分析。";
        }
        if (level === "UNKNOWN") {
            return "识别完成，但当前食物暂无完整营养库映射。";
        }
        if (level === "NONE") {
            return "未检测到有效食物，请重新拍照后再试。";
        }
        if (level === "ERROR") {
            return "识别失败，请检查上传内容或网络后重试。";
        }
        return "识别完成。";
    }

    function getReminderText(payload) {
        const analysis = payload.analysis || {};
        const mealSlot = analysis.meal_slot ? mealSlotLabel(analysis.meal_slot) : "";
        const riskRatioText = payload.risk_ratio ? `风险占比约 ${formatNumber(Number(payload.risk_ratio) * 100, 0)}%` : "";
        return [mealSlot, riskRatioText].filter(Boolean).join(" · ") || "等待生成慢病提醒";
    }

    function applyDetectionResult(payload) {
        latestResult = payload || {};
        hasScanned = true;

        if (payload.imgBase64) {
            setRemotePreview(payload.imgBase64);
        }

        setBadge(payload.level || "SAFE");
        conclusionText.textContent = getConclusionTitle(payload);
        reasonText.textContent = getReasonText(payload);
        recommendedGrams.textContent = Number(payload.recommendation_grams) > 0
            ? `${formatNumber(payload.recommendation_grams, 0)} g`
            : "-- g";
        portionHint.textContent = payload.portion || payload.visual_tip || "等待识别后显示分量建议";
        plateWeight.textContent = Number(payload.actual_weight || payload.nutrition?.actual_weight) > 0
            ? `${formatNumber(payload.actual_weight || payload.nutrition?.actual_weight, 0)} g`
            : "-- g";
        foodName.textContent = payload.food || "未识别";
        riskNutrient.textContent = payload.main_risk_reason || "--";
        diseaseReminder.textContent = getReminderText(payload);
        mealMode.textContent = payload.analysis && payload.analysis.meal_slot
            ? `${mealSlotLabel(payload.analysis.meal_slot)} · 真实识别`
            : "真实拍照识别";

        renderPriority(payload);
        renderDetails(payload);
        renderTips(payload);
        setBusy(false, getStatusText(payload));

        const level = String(payload.level || "").toUpperCase();
        if (level === "SAFE" || level === "CAUTION" || level === "AVOID" || level === "UNKNOWN") {
            autoSpeak(buildSpeakText(payload));
        }
    }

    function showRequestError(message) {
        hasScanned = true;
        latestResult = {
            level: "ERROR",
            food: "上传失败",
            tip: message
        };
        setBadge("ERROR");
        conclusionText.textContent = "上传或识别失败";
        reasonText.textContent = message;
        recommendedGrams.textContent = "-- g";
        portionHint.textContent = "请重新扫描";
        plateWeight.textContent = "-- g";
        foodName.textContent = "未完成识别";
        riskNutrient.textContent = "--";
        diseaseReminder.textContent = "请检查网络、图片格式或服务状态";
        mealMode.textContent = defaultMealMode;
        renderPlaceholderPriority("上传成功后将在这里显示本餐优先关注的营养项。");
        detailGrid.innerHTML = "";
        renderTips(latestResult);
        setBusy(false, "上传失败，请检查网络或稍后重试。");
    }

    async function uploadImage(file) {
        setBusy(true, "识别中，请稍候...");
        detailPanel.classList.remove("open");
        detailToggle.textContent = "查看详情";
        renderPlaceholderPriority("识别中，正在等待服务端返回真实分析结果。");
        detailGrid.innerHTML = "";
        setLocalPreview(file);

        const formData = new FormData();
        formData.append("image", file, file.name || "capture.jpg");
        formData.append("disease", config.diseaseKey);

        try {
            const response = await fetch(config.detectEndpoint, {
                method: "POST",
                body: formData
            });

            let payload = null;
            try {
                payload = await response.json();
            } catch (error) {
                throw new Error("服务返回格式异常，请稍后重试。");
            }

            if (!response.ok) {
                throw new Error(payload && (payload.tip || payload.message) || "识别请求失败，请稍后重试。");
            }

            applyDetectionResult(payload || {});
        } catch (error) {
            showRequestError(error.message || "上传失败，请稍后重试。");
        }
    }

    function openCameraCapture() {
        if (rescanBtn.disabled) return;
        fileInput.click();
    }

    function resetView() {
        latestResult = null;
        hasScanned = false;
        setBadge("SAFE");
        conclusionText.textContent = defaultTitle;
        reasonText.textContent = defaultReason;
        recommendedGrams.textContent = "-- g";
        portionHint.textContent = "等待识别后显示分量建议";
        plateWeight.textContent = "-- g";
        foodName.textContent = "等待识别菜品";
        riskNutrient.textContent = "--";
        diseaseReminder.textContent = "等待生成慢病提醒";
        mealMode.textContent = defaultMealMode || "真实拍照识别";
        detailPanel.classList.remove("open");
        detailToggle.textContent = "查看详情";
        renderPlaceholderPriority("点击下方按钮开始拍照识别。");
        detailGrid.innerHTML = "";
        renderTips({});
        setPreviewSource("", "");
        setBusy(false, defaultStatus);
    }

    window.speakResult = function () {
        if (!latestResult) return;
        autoSpeak(buildSpeakText(latestResult));
    };

    window.goBackToConfig = function () {
        autoSpeak(config.backSpeech || "返回输入页。");
        window.setTimeout(function () {
            window.location.href = config.inputPage;
        }, 420);
    };

    window.resetUI = function () {
        openCameraCapture();
    };

    detailToggle.addEventListener("click", function () {
        const isOpen = detailPanel.classList.toggle("open");
        detailToggle.textContent = isOpen ? "收起详情" : "查看详情";
    });

    fileInput.addEventListener("change", function (event) {
        const file = event.target.files && event.target.files[0];
        fileInput.value = "";
        if (!file) return;
        uploadImage(file);
    });

    resetView();
    autoSpeak(config.introSpeech || `${defaultPageTitle}页面已准备好，请点击下方按钮开始拍照识别。`);
})();
