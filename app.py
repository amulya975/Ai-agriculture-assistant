import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Google Gemini
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Agriculture Assistant",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f4fff4;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #16852a;
}

.subtitle {
    font-size: 19px;
    color: #555;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🌱 AI Agriculture Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart farming assistant for crop recommendation, '
    'plant disease detection, weather information and '
    'AI farming guidance.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# CROP RECOMMENDATION MODEL
# ============================================================

@st.cache_resource
def load_crop_model():

    data_file = "Crop_recommendation.csv"

    if not os.path.exists(data_file):
        return None, None, None

    data = pd.read_csv(data_file)

    required_columns = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "label"
    ]

    for column in required_columns:

        if column not in data.columns:
            raise ValueError(
                f"Missing column in CSV: {column}"
            )

    features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    X = data[features]

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        data["label"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = model.score(
        X_test,
        y_test
    )

    return model, encoder, accuracy



# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 Agriculture Assistant")

option = st.sidebar.radio(
    "Choose a feature",
    [
        "🏠 Home",
        "🌱 Crop Recommendation",
        "🌦️ Weather Information",
        "🤖 AI Farmer Chatbot"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    AI Agriculture Assistant

    Features:

    🌱 Crop Recommendation
    🍃 Disease Detection
    🌦️ Weather Information
    🤖 AI Farmer Chatbot
    """
)


# ============================================================
# HOME
# ============================================================

if option == "🏠 Home":

    st.header(
        "🌱 Welcome to AI Agriculture Assistant"
    )

    st.write(
        "An AI-powered agricultural decision-support "
        "system designed to help farmers with crop "
        "selection, plant health, weather information "
        "and farming questions."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">

            ### 🌱 Crop Recommendation

            Uses soil and environmental parameters:

            • Nitrogen  
            • Phosphorus  
            • Potassium  
            • Temperature  
            • Humidity  
            • Soil pH  
            • Rainfall  

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">

            ### 🌦️ Weather Information

            Provides:

            • Temperature  
            • Humidity  
            • Wind speed  
            • Rainfall  
            • Weather condition  
            • Farming guidance  

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            ### 🍃 Plant Disease Detection

            Upload a plant leaf image.

            The CNN model predicts a possible
            plant disease and confidence level.

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">

            ### 🤖 AI Farmer Chatbot

            Ask questions about:

            • Crops  
            • Soil  
            • Irrigation  
            • Fertilizers  
            • Pests  
            • Plant diseases  
            • Farming techniques  

            </div>
            """,
            unsafe_allow_html=True
        )

    st.success(
        "Select a feature from the sidebar to begin."
    )


# ============================================================
# CROP RECOMMENDATION
# ============================================================

elif option == "🌱 Crop Recommendation":

    st.header(
        "🌱 Crop Recommendation System"
    )

    st.write(
        "Enter soil and environmental values "
        "to get an ML-based crop recommendation."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌾 Soil Information"
        )

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=200.0,
            value=50.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

    with col2:

        st.subheader(
            "🌦️ Environmental Information"
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=25.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=100.0
        )

    st.divider()

    if st.button(
        "🌾 Recommend Best Crop",
        use_container_width=True
    ):

        try:

            model, encoder, accuracy = load_crop_model()

            if model is None:

                st.error(
                    "Crop recommendation file was not found."
                )

            else:

                input_data = pd.DataFrame(
                    [[
                        nitrogen,
                        phosphorus,
                        potassium,
                        temperature,
                        humidity,
                        ph,
                        rainfall
                    ]],
                    columns=[
                        "N",
                        "P",
                        "K",
                        "temperature",
                        "humidity",
                        "ph",
                        "rainfall"
                    ]
                )

                prediction = model.predict(
                    input_data
                )

                crop = encoder.inverse_transform(
                    prediction
                )[0]

                st.success(
                    f"🌱 Recommended Crop: "
                    f"**{crop.upper()}**"
                )

                st.info(
                    f"Model validation accuracy: "
                    f"**{accuracy * 100:.2f}%**"
                )

        except Exception as e:

            st.error(
                f"Crop recommendation error: {e}"
            )


# ============================================================
# PLANT DISEASE DETECTION
# ============================================================

elif option == "🍃 Plant Disease Detection":

    st.header(
        "🍃 Plant Disease Detection"
    )

    st.write(
        "Upload a clear plant leaf image "
        "to detect a possible disease."
    )

    st.warning(
        "This is a preliminary AI screening and "
        "should not replace professional agricultural advice."
    )

    uploaded_file = st.file_uploader(
        "📷 Upload Plant Leaf Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Plant Disease",
            use_container_width=True
        ):

            try:

                model = load_disease_model()

                if model is None:

                    st.error(
                        "plant_disease_model.h5 was not found."
                    )

                    st.info(
                        "Place your trained model in "
                        "the project folder."
                    )

                else:

                    disease, confidence = predict_disease(
                        model,
                        image
                    )

                    st.subheader(
                        "🔎 Detection Result"
                    )

                    clean_name = disease.replace(
                        "___",
                        " - "
                    ).replace(
                        "_",
                        " "
                    )

                    st.success(
                        f"Possible condition: "
                        f"**{clean_name}**"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    if confidence < 0.60:

                        st.warning(
                            "Low model confidence. "
                            "Try a clear image with the "
                            "leaf filling most of the frame."
                        )

                    elif "healthy" in disease.lower():

                        st.success(
                            "🌿 The plant appears healthy "
                            "according to the model."
                        )

                    else:

                        st.warning(
                            "⚠️ Possible disease detected. "
                            "Consult an agricultural expert "
                            "before treatment."
                        )

            except Exception as e:

                st.error(
                    f"Disease detection error: {e}"
                )


# ============================================================
# WEATHER INFORMATION
# ============================================================

elif option == "🌦️ Weather Information":

    st.header(
        "🌦️ Weather Information"
    )

    st.write(
        "Check current weather conditions "
        "and understand their possible effect "
        "on farming."
    )

    st.divider()

    api_key = os.getenv(
        "OPENWEATHER_API_KEY"
    )

    if not api_key:

        st.error(
            "OPENWEATHER_API_KEY is not configured."
        )

        st.info(
            "Add OPENWEATHER_API_KEY to your .env file."
        )

    else:

        city = st.text_input(
            "📍 Enter City",
            value="Hyderabad"
        )

        if st.button(
            "🌦️ Get Weather",
            use_container_width=True
        ):

            if not city.strip():

                st.warning(
                    "Please enter a city name."
                )

            else:

                try:

                    url = (
                        "https://api.openweathermap.org/data/2.5/weather"
                    )

                    params = {
                        "q": city,
                        "appid": api_key,
                        "units": "metric"
                    }

                    response = requests.get(
                        url,
                        params=params,
                        timeout=10
                    )

                    if response.status_code == 200:

                        weather_data = response.json()

                        temperature = weather_data[
                            "main"
                        ]["temp"]

                        feels_like = weather_data[
                            "main"
                        ]["feels_like"]

                        humidity = weather_data[
                            "main"
                        ]["humidity"]

                        pressure = weather_data[
                            "main"
                        ]["pressure"]

                        wind_speed = weather_data[
                            "wind"
                        ]["speed"]

                        rainfall = weather_data.get(
                            "rain",
                            {}
                        ).get(
                            "1h",
                            0
                        )

                        condition = weather_data[
                            "weather"
                        ][0]["description"]

                        country = weather_data[
                            "sys"
                        ]["country"]

                        st.success(
                            f"Weather for "
                            f"**{city.title()}, {country}**"
                        )

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:

                            st.metric(
                                "🌡️ Temperature",
                                f"{temperature:.1f} °C"
                            )

                        with col2:

                            st.metric(
                                "💧 Humidity",
                                f"{humidity}%"
                            )

                        with col3:

                            st.metric(
                                "💨 Wind",
                                f"{wind_speed:.1f} m/s"
                            )

                        with col4:

                            st.metric(
                                "🌧️ Rainfall",
                                f"{rainfall:.1f} mm"
                            )

                        st.divider()

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.write(
                                f"**Condition:** "
                                f"{condition.title()}"
                            )

                        with col2:

                            st.write(
                                f"**Feels Like:** "
                                f"{feels_like:.1f} °C"
                            )

                        with col3:

                            st.write(
                                f"**Pressure:** "
                                f"{pressure} hPa"
                            )

                        st.divider()

                        st.subheader(
                            "🌾 Farming Guidance"
                        )

                        if rainfall > 5:

                            st.info(
                                "🌧️ Significant rainfall detected. "
                                "Avoid unnecessary irrigation "
                                "and monitor fields for excess water."
                            )

                        elif humidity > 80:

                            st.warning(
                                "💧 High humidity. Monitor crops "
                                "for possible fungal problems."
                            )

                        elif temperature > 35:

                            st.warning(
                                "🔥 High temperature. Ensure "
                                "adequate water availability "
                                "and monitor heat-sensitive crops."
                            )

                        elif temperature < 10:

                            st.warning(
                                "❄️ Low temperature. Protect "
                                "temperature-sensitive crops."
                            )

                        else:

                            st.success(
                                "🌱 Current conditions are "
                                "moderate. Continue monitoring "
                                "soil moisture and crop health."
                            )

                    else:

                        st.error(
                            "Could not find weather information. "
                            "Check the city name and API key."
                        )

                except requests.exceptions.Timeout:

                    st.error(
                        "Weather service timed out. "
                        "Please try again."
                    )

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Network error: {e}"
                    )

                except Exception as e:

                    st.error(
                        f"Weather error: {e}"
                    )


# ============================================================
# AI FARMER CHATBOT - GOOGLE GEMINI
# ============================================================

elif option == "🤖 AI Farmer Chatbot":

    st.header(
        "🤖 AI Farmer Chatbot"
    )

    st.write(
        "Ask questions about crops, soil, irrigation, "
        "fertilizers, pests, diseases and farming."
    )

    # --------------------------------------------------------
    # GEMINI API KEY
    # --------------------------------------------------------

    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not gemini_api_key:

        st.error(
            "GEMINI_API_KEY is not configured."
        )

        st.info(
            "Add GEMINI_API_KEY to your .env file."
        )

    else:

        # ----------------------------------------------------
        # CREATE GEMINI CLIENT
        # ----------------------------------------------------

        try:

            client = genai.Client(
                api_key=gemini_api_key
            )

            # ------------------------------------------------
            # CHAT HISTORY
            # ------------------------------------------------

            if "messages" not in st.session_state:

                st.session_state.messages = []

            # ------------------------------------------------
            # DISPLAY PREVIOUS MESSAGES
            # ------------------------------------------------

            for message in st.session_state.messages:

                with st.chat_message(
                    message["role"]
                ):

                    st.markdown(
                        message["content"]
                    )

            # ------------------------------------------------
            # CHAT INPUT
            # ------------------------------------------------

            prompt = st.chat_input(
                "Ask your farming question..."
            )

            if prompt:

                # --------------------------------------------
                # SAVE USER MESSAGE
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt
                    }
                )

                with st.chat_message("user"):

                    st.markdown(
                        prompt
                    )

                # --------------------------------------------
                # SYSTEM INSTRUCTION
                # --------------------------------------------

                system_message = """
You are an AI Farmer Assistant.

Your purpose is to help farmers with:

1. Crop selection
2. Soil management
3. Irrigation
4. Fertilizers
5. Plant diseases
6. Pest management
7. Farming techniques
8. Weather-related farming decisions
9. General agriculture questions

Rules:

- Use simple language.
- Give practical and safe information.
- Do not claim certainty about disease diagnosis.
- Do not recommend dangerous chemical use.
- For pesticide or serious crop-loss problems,
  recommend consulting a qualified agricultural expert.
- If the question is unrelated to agriculture,
  politely explain that you are an agriculture assistant.
- Prefer concise, actionable answers.
- When useful, organize the answer with bullet points.
"""

                # --------------------------------------------
                # BUILD CONVERSATION FOR GEMINI
                # --------------------------------------------

                conversation = ""

                for message in st.session_state.messages:

                    role = message["role"]

                    if role == "user":

                        conversation += (
                            f"\nFarmer: {message['content']}"
                        )

                    elif role == "assistant":

                        conversation += (
                            f"\nAssistant: {message['content']}"
                        )

                full_prompt = f"""
{system_message}

Conversation history:
{conversation}

Now answer the farmer's latest question.
"""

                # --------------------------------------------
                # GENERATE GEMINI RESPONSE
                # --------------------------------------------

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Gemini is thinking..."
                    ):

                        try:

                            response = client.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=full_prompt,
                                config=types.GenerateContentConfig(
                                    temperature=0.4,
                                    max_output_tokens=1000
                                )
                            )

                            answer = response.text

                            if not answer:

                                answer = (
                                    "Sorry, I could not generate "
                                    "a response. Please try again."
                                )

                            st.markdown(
                                answer
                            )

                            # -------------------------------
                            # SAVE ASSISTANT RESPONSE
                            # -------------------------------

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": answer
                                }
                            )

                        except Exception as e:

                            st.error(
                                f"Gemini chatbot error: {e}"
                            )

        except Exception as e:

            st.error(
                f"Could not initialize Gemini: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 AI Agriculture Assistant | "
    "Crop Recommendation + Plant Disease Detection + "
    "Weather Information + Gemini AI Farmer Chatbot"
)
