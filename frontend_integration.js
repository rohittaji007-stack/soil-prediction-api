// ============================================================
// NILDURGA SOFTWARE INTEGRATION — HONEST VERSION
// Complete drop-in replacement for your device script.
// Reports ONLY what the training data supports:
//   N       -> numeric value        (reliability: measured)
//   pH,Fe,Cu-> Low/Medium/High       (reliability: screening)
//   others  -> "Not available"       (no reliable signal — never faked)
// No hardcoded calibration multipliers.
// ============================================================

const SUPABASE_URL = "https://gzwhkukpqdjmkokwazec.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd6d2hrdWtwcWRqbWtva3dhemVjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUxNzI5NTEsImV4cCI6MjA5MDc0ODk1MX0.my6LHtTE_tbSm1vrx8tyV4PkWx2cugjmvWRdWHSIyGk";

window.currentSampleId = null;

// short key -> { value widget, conclusion/note widget }
const WIDGETS = {
  N:  { val: "43pd721fx", note: "7ns1xvmn8" },
  P:  { val: "9goxm8xs6", note: "rbkicsq7g" },
  K:  { val: "ylz0szyhd", note: "tlepjg3bz" },
  OC: { val: "7yfbdxi7v", note: "jka2l4p44" },
  PH: { val: "xbrbcjgk8", note: "crikour6q" },
  EC: { val: "9mh5rmj0e", note: "4b8s342p4" },
  FE: { val: "cjubgjfvx", note: "zfimt5qnv" },
  MN: { val: "j8jy8rc91", note: "fq2e567ih" },
  CU: { val: "q3qsj5rgv", note: "qmubss369" },
  ZN: { val: "ees36m8sh", note: "hgj6vptxf" },
  B:  { val: "wj5xkhfyi", note: "4ykllmh0t" },
  S:  { val: "2nr6ols8q", note: "87zdxbonv" },
};

// --- FETCH SAMPLE DETAILS FROM BACKEND ---
async function fetchSampleDetails(sampleId) {
  const url = `${SUPABASE_URL}/functions/v1/get-sample?sample_id=${encodeURIComponent(sampleId)}`;
  console.log("Fetching sample details for:", sampleId);
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { "Content-Type": "application/json", apikey: SUPABASE_ANON_KEY },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { sample } = await response.json();
    console.log("Sample details received:", JSON.stringify(sample, null, 2));
    window.currentSampleId = sampleId;
    if (widget) {
      widget.setValue("7h62weags", sample.farmer_name || "");
      widget.setValue("dckudnppc", sample.farmer_phone || "");
      widget.setValue("60ptjrkhk", sample.farm_location || "");
      widget.setValue("ar97yxdif", sample.crop_type || "");
      widget.setValue("yqt123f6p", sample.land_survey_number || "");
      widget.setValue("cj9rnuw8g", sample.subdivision || "");
      widget.setValue("6j2443i00", sample.district || "");
      widget.setValue("6uf19sq9b", sample.collector_name || "");
    }
    return sample;
  } catch (error) {
    console.error("Failed to fetch sample:", error);
    if (widget) widget.setValue("widget_1767974080573", "Error: Sample not found");
    return null;
  }
}

// --- RENDER HONEST REPORT ---
function renderReport(report, unavailable) {
  if (!widget) return;
  for (const [key, item] of Object.entries(report)) {
    const w = WIDGETS[key];
    if (!w) continue;
    if (item.type === "value") {
      widget.setValue(w.val, `${item.value} ${item.unit}`.trim());
      widget.setValue(w.note, `Measured (R²=${item.cv_r2})`);
    } else if (item.type === "class") {
      widget.setValue(w.val, item.class);
      widget.setValue(w.note, `Screening (${Math.round(item.confidence * 100)}% conf)`);
    }
  }
  for (const key of unavailable || []) {
    const w = WIDGETS[key];
    if (!w) continue;
    widget.setValue(w.val, "N/A");
    widget.setValue(w.note, "Not available");
  }
}

// --- SUBMIT ONLY TRUSTWORTHY RESULTS ---
// NOTE: your Supabase `submit-results` edge function must accept this shape.
async function submitResultsToBackend(sampleId, apiResult) {
  const url = `${SUPABASE_URL}/functions/v1/submit-results`;
  const results = {};
  for (const [key, item] of Object.entries(apiResult.final_report)) {
    if (item.type === "value") {
      results[key] = { value: item.value, unit: item.unit, reliability: "measured" };
    } else if (item.type === "class") {
      results[key] = { level: item.class, confidence: item.confidence, reliability: "screening" };
    }
  }
  const payload = {
    sample_id: sampleId,
    results,
    unavailable: apiResult.unavailable,
    disclaimer: apiResult.disclaimer,
  };
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", apikey: SUPABASE_ANON_KEY },
      body: JSON.stringify(payload),
    });
    const out = await response.json();
    if (!response.ok) throw new Error(out.error || `HTTP ${response.status}`);
    console.log("Results submitted:", out);
    if (widget) widget.setValue("widget_1767974080573", "✅ Results saved!");
    return out;
  } catch (error) {
    console.error("Failed to submit results:", error);
    if (widget) widget.setValue("widget_1767974080573", "❌ Upload failed");
    return null;
  }
}

// ============================================================
// IOT LOGIC
// ============================================================
const sensorId = context.deviceId || "your_sensor_id_here";

if (!window.sensorBatchBuffer) {
  window.sensorBatchBuffer = [];
  console.log("Initialized new sensor buffer.");
}

function processIncomingReading(newReading) {
  if (!Array.isArray(newReading) || newReading.length !== 18) {
    console.error("Error: Input data must be an array of 18 values.", newReading);
    return;
  }
  window.sensorBatchBuffer.push(newReading);
  const currentCount = window.sensorBatchBuffer.length;
  console.log(`Reading added. Buffer: ${currentCount}/10`);
  if (widget) widget.setValue("widget_1767974080573", currentCount + "/10");
  if (currentCount >= 10) {
    console.log("Buffer full. Triggering API...");
    const batchData = [...window.sensorBatchBuffer];
    window.sensorBatchBuffer = [];
    predictSoilNutrients(sensorId, batchData);
  }
}

async function predictSoilNutrients(deviceSensorId, data) {
  const url = `https://soil-prediction-api.onrender.com/predict_batch/${deviceSensorId}`;
  console.log("Calling Soil Prediction API...");
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    console.log("API Response:", JSON.stringify(result, null, 2));

    if (result.status === "success" && result.final_report) {
      renderReport(result.final_report, result.unavailable);
      if (window.currentSampleId) {
        await submitResultsToBackend(window.currentSampleId, result);
      } else {
        console.warn("No sample loaded — results shown locally only.");
      }
    } else if (result.status === "error") {
      console.error("API error:", result.message);
      if (widget) widget.setValue("widget_1767974080573", "❌ " + result.message);
    }
    return result;
  } catch (error) {
    console.error("API call failed:", error);
    throw error;
  }
}

// --- WEBSOCKET LISTENER ---
ws.onMessage((data) => {
  console.log("[Script] Incoming WebSocket data:", data);
  if (data.payload?.widgetId) {
    if (data.payload.widgetId == "widget_1767939405303") {
      processIncomingReading(data.payload.value);
    } else {
      console.log("id not match " + JSON.stringify(data.payload));
    }
  }
});

// --- TRANSFER BUTTON ---
widget.on("widget_1767974315613", "click", () => {
  console.log("Transfer clicked");
  widget.setValue("7h62weags", widget.getValue("widget_1770091660844"));
  widget.setValue("dckudnppc", widget.getValue("ns89orq6m"));
  widget.setValue("60ptjrkhk", widget.getValue("w37h05as3"));
  widget.setValue("yqt123f6p", widget.getValue("b0hn820xy"));
  widget.setValue("cj9rnuw8g", widget.getValue("nhfv1ttwx"));
  widget.setValue("6j2443i00", widget.getValue("77nuf3o3c"));
  widget.setValue("ar97yxdif", widget.getValue("8gg0i3yar"));
  widget.setValue("6uf19sq9b", widget.getValue("m88u2ehq1"));
});

// --- LOAD SAMPLE BY ID (hook to your QR scan widget) ---
// widget.on("YOUR_QR_SCAN_WIDGET_ID", "change", async (sampleId) => {
//   await fetchSampleDetails(sampleId);
// });
