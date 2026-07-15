// ============================================================
// NILDURGA SOFTWARE INTEGRATION — FULL REPORT (honest confidence)
// Complete drop-in replacement for your device script.
//
// Shows a numeric value for ALL 12 parameters, plus a Low/Med/High/Normal
// conclusion vs the required range, plus a confidence marker:
//   high     -> real signal (N)
//   moderate -> coarse screening (pH, Fe, Cu)
//   low      -> indicative estimate (shown with "· est.")
// Numbers come straight from the model — no fabricated / fudge multipliers.
// ============================================================

const SUPABASE_URL = "https://gzwhkukpqdjmkokwazec.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd6d2hrdWtwcWRqbWtva3dhemVjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUxNzI5NTEsImV4cCI6MjA5MDc0ODk1MX0.my6LHtTE_tbSm1vrx8tyV4PkWx2cugjmvWRdWHSIyGk";

window.currentSampleId = null;

// short key -> { val widget, note/conclusion widget }
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

// --- STATUS HELPERS (vs the required range for each parameter) ---
function bandStatus(value, min, max) {
  const v = parseFloat(value);
  if (isNaN(v)) return "-";
  if (v < min) return "Low";
  if (v > max) return "High";
  return "Medium";
}
function greaterThanStatus(value, limit) {
  const v = parseFloat(value);
  if (isNaN(v)) return "-";
  return v > limit ? "Normal" : "Low";
}
function phStatus(ph) {
  const v = parseFloat(ph);
  if (isNaN(v)) return "-";
  if (v < 6.5) return "Acidic";
  if (v < 7.0) return "Normal";
  if (v < 8.5) return "Alkaline";
  return "Highly Alkaline";
}

// short key -> function that turns a value into a conclusion string
const STATUS = {
  N:  (v) => bandStatus(v, 280, 499.99),
  P:  (v) => bandStatus(v, 14, 20.99),
  K:  (v) => bandStatus(v, 150, 199.99),
  OC: (v) => bandStatus(v, 0.4, 0.59),
  PH: (v) => phStatus(v),
  EC: (v) => bandStatus(v, 0, 1),
  FE: (v) => bandStatus(v, 4.5, 18),
  MN: (v) => bandStatus(v, 2, 8),
  CU: (v) => bandStatus(v, 0.2, 0.8),
  ZN: (v) => bandStatus(v, 0.6, 18),
  B:  (v) => greaterThanStatus(v, 0.5),
  S:  (v) => greaterThanStatus(v, 10),
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

// --- RENDER FULL REPORT ---
function renderReport(report) {
  if (!widget) return;
  for (const [key, item] of Object.entries(report)) {
    const w = WIDGETS[key];
    if (!w) continue;

    // Result column: number + unit
    widget.setValue(w.val, `${item.value}${item.unit ? " " + item.unit : ""}`);

    // Conclusion column: status vs range + confidence marker
    const status = STATUS[key] ? STATUS[key](item.value) : "-";
    const marker = item.confidence === "low" ? " · est." : "";
    widget.setValue(w.note, `${status}${marker}`);
  }
}

// --- SUBMIT RESULTS ---
async function submitResultsToBackend(sampleId, apiResult) {
  const url = `${SUPABASE_URL}/functions/v1/submit-results`;
  const results = {};
  for (const [key, item] of Object.entries(apiResult.final_report)) {
    results[key] = {
      value: item.value,
      unit: item.unit,
      confidence: item.confidence,
      status: STATUS[key] ? STATUS[key](item.value) : null,
    };
    if (item.class) results[key].level = item.class;
  }
  const payload = {
    sample_id: sampleId,
    results,
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
  if (widget) widget.setValue("widget_1767974080573", currentCount + "/10");
  if (currentCount >= 10) {
    const batchData = [...window.sensorBatchBuffer];
    window.sensorBatchBuffer = [];
    predictSoilNutrients(sensorId, batchData);
  }
}

async function predictSoilNutrients(deviceSensorId, data) {
  const url = `https://soil-prediction-api.onrender.com/predict_batch/${deviceSensorId}`;
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
      renderReport(result.final_report);
      if (window.currentSampleId) {
        await submitResultsToBackend(window.currentSampleId, result);
      } else {
        console.warn("No sample loaded — results shown locally only.");
      }
    } else if (result.status === "error") {
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
