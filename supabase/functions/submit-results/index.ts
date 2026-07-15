import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// HONEST result shape sent by frontend_integration.js.
// Each parameter is EITHER a measured numeric value OR a screening class.
type MeasuredResult = { value: number; unit: string; reliability: "measured" };
type ScreeningResult = { level: string; confidence: number; reliability: "screening" };

interface TestResults {
  sample_id: string;
  results: Record<string, MeasuredResult | ScreeningResult>;
  unavailable?: string[];   // params with no reliable signal (not predicted)
  disclaimer?: string;
}

// Short keys the API may send.
const VALID_KEYS = ["N", "P", "K", "OC", "PH", "EC", "FE", "MN", "CU", "ZN", "B", "S"];

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  try {
    const body: TestResults = await req.json();

    if (!body.sample_id || !body.results || typeof body.results !== "object") {
      return new Response(JSON.stringify({ error: "sample_id and results are required" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Keep only recognised parameter keys; drop anything unexpected.
    const cleanResults: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(body.results)) {
      if (VALID_KEYS.includes(k)) cleanResults[k] = v;
    }
    if (Object.keys(cleanResults).length === 0) {
      return new Response(JSON.stringify({ error: "no valid result parameters supplied" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Verify sample exists
    const { data: sample, error: fetchError } = await supabase
      .from("soil_samples")
      .select("id, status")
      .eq("sample_id", body.sample_id)
      .single();

    if (fetchError || !sample) {
      return new Response(JSON.stringify({ error: "Sample not found" }), {
        status: 404,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Store the full honest report in the existing JSONB column.
    // Nested so nothing is lost: which params were measured vs screening vs
    // unavailable, plus the disclaimer. The reader UI should read
    // test_results.results.<KEY>.value / .level.
    const testResults = {
      results: cleanResults,
      unavailable: Array.isArray(body.unavailable) ? body.unavailable : [],
      disclaimer: body.disclaimer ?? null,
      reported_at: new Date().toISOString(),
    };

    const { error: updateError } = await supabase
      .from("soil_samples")
      .update({
        test_results: testResults,
        status: "completed",
        result_added_at: new Date().toISOString(),
        testing_date: new Date().toISOString().split("T")[0],
      })
      .eq("id", sample.id);

    if (updateError) {
      return new Response(JSON.stringify({ error: updateError.message }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(JSON.stringify({
      success: true,
      message: "Test results submitted successfully",
      sample_id: body.sample_id,
    }), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("submit-results error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
