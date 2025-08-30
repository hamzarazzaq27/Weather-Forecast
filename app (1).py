"""
app.py - Gradio weather forecast + HF summarizer

How it works:
1. Geocodes the user-entered location using Open-Meteo geocoding API.
2. Fetches daily & hourly weather forecast from Open-Meteo.
3. (Optional) Calls Hugging Face text-generation inference API to produce
   a short human-friendly forecast summary. Set HF_API_TOKEN as an env var.
4. Shows results in a Gradio web UI.

Run:
    pip install -r requirements.txt
    export HF_API_TOKEN="hf_..."   # optional, for Hugging Face summarization
    python app.py
"""

import os
import requests
import datetime
from typing import Tuple, Dict, Any, Optional
import gradio as gr

# -------- Config ----------
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/"  # append model id
HF_MODEL = "google/flan-t5-small"  # small model recommended for quick summaries (change if you like)
# --------------------------

def geocode_place(place: str) -> Optional[Dict[str, Any]]:
    """Return the top geocoding result {name, latitude, longitude, country, timezone} or None."""
    params = {"name": place, "count": 1, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("results"):
        return data["results"][0]
    return None

def fetch_open_meteo(lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
    """Fetch daily/hourly forecast from Open-Meteo for the next `days` days."""
    start_date = datetime.date.today()
    end_date = start_date + datetime.timedelta(days=days-1)
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,sunrise,sunset",
        "hourly": "temperature_2m,relativehumidity_2m,precipitation,winddirection_10m,windspeed_10m",
        "timezone": "auto",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    r = requests.get(FORECAST_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def format_forecast_text(place_name: str, forecast_json: Dict[str, Any]) -> str:
    """Create a plain-language summary from the Open-Meteo daily forecast data."""
    daily = forecast_json.get("daily", {})
    dates = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    weathercodes = daily.get("weathercode", [])

    lines = [f"Weather forecast for {place_name}:\n"]
    for d, hi, lo, p, wc in zip(dates, tmax, tmin, precip, weathercodes):
        date = datetime.datetime.fromisoformat(d).strftime("%a %d %b")
        precip_note = "no significant rain" if (p is None or p == 0) else f"{p} mm precipitation expected"
        # Basic interpretation of weathercode (simple)
        wc_note = _weathercode_to_text(wc)
        lines.append(f"{date}: {wc_note}. Temp {int(lo)}°C–{int(hi)}°C, {precip_note}.")
    return "\n".join(lines)

def _weathercode_to_text(code: int) -> str:
    """Very small mapping for Open-Meteo weathercode to text."""
    if code in (0,):
        return "Clear sky"
    if code in (1,2,3):
        return "Mainly clear to partly cloudy"
    if code in (45,48):
        return "Fog or depositing rime fog"
    if code in (51,53,55):
        return "Drizzle"
    if code in (61,63,65):
        return "Rain"
    if code in (71,73,75):
        return "Snow"
    if code in (95,96,99):
        return "Thunderstorms"
    return "Mixed/unknown weather"

def hf_summarize(text: str, model: str = HF_MODEL, hf_token: Optional[str] = None, max_length: int = 200) -> str:
    """Call HF inference API text2text-generation to summarize/rewritten text.
    Requires HF_API_TOKEN in env or passed as hf_token. Returns generated text or original on failure."""
    token = hf_token or os.environ.get("HF_API_TOKEN")
    if not token:
        return "(Hugging Face token not provided; summarization skipped)\n\n" + text[:2000]

    url = HF_INFERENCE_URL + model
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    payload = {
        "inputs": f"Summarize the following weather report in a short, user-friendly paragraph:\n\n{text}",
        "parameters": {"max_new_tokens": 120, "temperature": 0.1},
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # the inference API may return a list or object; handle both common cases
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        # sometimes models return text directly
        if isinstance(data, str):
            return data.strip()
        return "(Summarization produced unexpected output) " + str(data)[:1000]
    except Exception as e:
        return f"(Hugging Face summarization failed: {e})\n\n" + text[:2000]

# -------- Gradio interface logic --------
def get_forecast_for_place(place: str, days: int = 7, use_hf: bool = True) -> Tuple[str, Dict[str, Any]]:
    place = place.strip()
    if not place:
        return "Please enter a location (e.g., 'Rahim Yar Khan, Pakistan').", {}

    # 1) Geocode
    try:
        geo = geocode_place(place)
    except Exception as e:
        return f"Error during geocoding: {e}", {}

    if not geo:
        return f"Location not found: {place}", {}

    name = f"{geo.get('name')}, {geo.get('country')}"
    lat = geo.get("latitude")
    lon = geo.get("longitude")

    # 2) Fetch forecast
    try:
        forecast = fetch_open_meteo(lat, lon, days=int(days))
    except Exception as e:
        return f"Error fetching forecast for {name}: {e}", {}

    # 3) Build plain text summary
    raw_text_summary = format_forecast_text(name, forecast)

    # 4) (Optional) Hugging Face summarization
    hf_token = os.environ.get("HF_API_TOKEN")
    if use_hf and hf_token:
        summary = hf_summarize(raw_text_summary, hf_token=hf_token)
    elif use_hf and not hf_token:
        summary = "(HF token not set — set HF_API_TOKEN to enable model summarization)\n\n" + raw_text_summary
    else:
        summary = raw_text_summary

    # Return: friendly summary and raw JSON (so UI can let user view raw data)
    return summary, forecast

# -------- Gradio UI --------
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("## Pakistan Region Weather Forecast — Gradio + Hugging Face\nEnter a city/region in Pakistan (eg. `Rahim Yar Khan, Pakistan`) and select days.")
    with gr.Row():
        place_in = gr.Textbox(label="City / Region", placeholder="e.g., Rahim Yar Khan, Pakistan", lines=1)
        days_in = gr.Slider(minimum=1, maximum=14, step=1, value=7, label="Days of forecast")
    hf_checkbox = gr.Checkbox(value=True, label="Enable Hugging Face summary (requires HF_API_TOKEN env var)")
    get_btn = gr.Button("Get Forecast")

    with gr.Row():
        summary_out = gr.Textbox(label="Forecast Summary (human-friendly)", lines=8)
        raw_out = gr.JSON(label="Raw Forecast JSON")

    examples = gr.Examples(
        examples=[["Rahim Yar Khan, Pakistan", 7], ["Lahore, Pakistan", 5], ["Islamabad, Pakistan", 3]],
        inputs=[place_in, days_in],
    )

    def _action(place, days, use_hf_flag):
        text, raw = get_forecast_for_place(place, days, use_hf_flag)
        return text, raw

    get_btn.click(_action, inputs=[place_in, days_in, hf_checkbox], outputs=[summary_out, raw_out])

    gr.Markdown("**Notes:**\n- Open-Meteo is used for actual meteorological data (no API key required).\n- To enable better natural-language summaries, export your Hugging Face token: `export HF_API_TOKEN='hf_...'`.\n- Change `HF_MODEL` near the top to try different HF models (beware larger models -> slower / costlier).")

if __name__ == "__main__":
    demo.launch()
