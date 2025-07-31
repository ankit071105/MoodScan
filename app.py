import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import time
import joblib

from recommender_engine import generate_suggestions



# Load the model
with open("./Models/LifestylePovertyIndex2.pkl", "rb") as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(layout="wide")

# Load external styles from styles.css
with open("./styles/styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Lifestyle mapping
lifestyle_options = [
    "Athlete", "Early Riser", "Energy Drink Addict", "Fast Food Lover", "Gym Goer",
    "Healthy Eater", "Night Owl", "Non-Smoker", "Occasional Drinker",
    "Sedentary", "Smoker", "Yoga Enthusiast"
]
lifestyle_map = {val: i for i, val in enumerate(lifestyle_options)}

# Suggestion logic
def get_suggestion(score):
    if score < 2:
        return "⚠️ High-risk burnout zone; urgent need for recovery and rest."
    elif score < 5:
        return "🌀 Turbulent Zone — Fatigue and emotional strain accumulating."
    elif score < 7:
        return "⚖️ Functionally stressed — Managing, but taxed."
    elif score < 9:
        return "🌿 Resilient & Adaptive — Healthy flow."
    else:
        return "✨ Peak Harmony — Optimal alignment."

# Reusable display functions
def display_card(label, value):
    formatted = f"{value:.2f}" if value is not None else "N/A"
    st.markdown(f"""
        <div class="hover-card">
            <strong>{label}</strong> {formatted}
        </div>
    """, unsafe_allow_html=True)


def display_suggestion_card(title, score):
    message = get_suggestion(score)
    st.markdown(f"""
        <div class="small-hover-card">
            <span class="suggestion-label">{title}</span>
            {message}
        </div>
    """, unsafe_allow_html=True)

def display_resource_card(title, url, icon="🔗"):
    return f"""
    <div class="resource-card">
        <a href="{url}" target="_blank">{icon} <strong>{title}</strong></a>
    </div>
    """

def display_about_card():
    st.markdown(f"""
        <div class="about-card">
            <h3>📖 About</h3>
            <p>This platform empowers individuals to uncover the subtle yet powerful connections between their everyday routines and overall wellbeing. Through intelligent analysis of lifestyle data—such as sleep patterns, screen exposure, physical activity, and emotional states—it offers a personalized lens into the factors influencing stress levels, burnout risk, and mental resilience.

✨ Key Features:

Data-driven insights into burnout, stress balance, and wellness dynamics

Visual explanations that clearly link behaviors to wellbeing outcomes

Anonymous comparisons with national and peer benchmarks to contextualize scores

Curated resources and recommendations aligned with individual lifestyle trends

By transforming personal patterns into actionable knowledge, the platform aims to spark awareness, inspire healthier choices, and support users in building sustainable, balanced lifestyles—one mindful step at a time.</p>
        </div>
    """, unsafe_allow_html=True)

def display_story_card():
    st.markdown(f"""
        <div class="about-card">
            <h3>The Story</h3>
            <p>This began as more than an idea — it was a feeling. A quiet knowing that our daily habits shape more than our days; they shape our lives. That’s what sparked this journey.</p> <p>Every click, every feature, every word here carries a little heartbeat — born from late-night thoughts, real conversations, and the hope that we could make wellbeing more personal, more human, more kind.</p> <p>This isn’t just an app. It’s a gentle nudge. A reflection of the belief that small shifts can lead to brighter days — that understanding yourself is the first step toward taking care of yourself.</p> <p>What you see here is built with care, intention, and joy — to help you pause, reflect, and grow. It’s for the student pulling all-nighters, the dreamer chasing balance, the soul seeking calm in the chaos.</p> <p>We’re still learning, still building, still dreaming. But through it all, one thing stays true: this was made with heart — for every moment you choose to take care of yours. 💙</p>
              
         </div>
    """, unsafe_allow_html=True)

def display_intro_card():
    st.markdown("""
        <div id="top" class="card" style="display:flex; justify-content:space-between; align-items:center; padding: 6px 10px; margin-bottom: 10px;">
            <p style="margin: 0; font-size: 14px; color:#002f4b;">🧭 Explore your lifestyle’s impact on burnout, wellbeing & productivity</p>
            <div>
                <a href="#about" style="color:#002f4b; text-decoration:none; font-size: 13px;">About</a> &nbsp;&nbsp;|&nbsp;&nbsp; 
                <a href="#story" style="color:#002f4b; text-decoration:none; font-size: 13px;">The Story</a>
            </div>
        </div>
    """, unsafe_allow_html=True)



display_intro_card()

# Title card
st.markdown("""
<div class="card" style="text-align:center;">
<h1>🌿 Lifestyle Wellness & Burnout Risk Assessment</h1>

</div>
""", unsafe_allow_html=True)

# Top Navigation Card

# Input Form
with st.container():
    st.markdown("""<div class="card"><h3>📥 Enter Your Lifestyle & Routine Data</h3></div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.3, 1.3])
    age = col1.number_input("Age", min_value=10.0, max_value=60.0, value=20.0, step=0.1)
    sleep_hours = col2.number_input("Sleep Hours", 0.0, 10.0, 6.5, 0.1)
    study_hours = col3.number_input("Study Hours", 0.0, 10.0, 4.0, 0.1)

    col4, col5 = st.columns(2)
    screen_time = col4.number_input("Screen Time Post 10PM", 0.0, 10.0, 2.0, 0.1)
    physical_activity = col5.number_input("Physical Activity Minutes", 0.0, 300.0, 30.0, 1.0)

    col6, col7 = st.columns(2)
    mood_score = col6.number_input("Mood Score", 0.0, 10.0, 6.8, 0.1)
    sleep_debt = col7.number_input("Sleep Debt", 0.0, 10.0, 1.2, 0.1)

    selected_habits = st.multiselect("Lifestyle Type (select one or more)", options=lifestyle_options)
    lifestyle_encoded = lifestyle_map[selected_habits[0]] if selected_habits else 0

    predict_btn = st.button("🔮 Predict Wellness & Burnout Scores")

# Initialize predictions
if "predictions" not in st.session_state:
    st.session_state.predictions = None

if predict_btn:
    input_data = np.array([[age, sleep_hours, study_hours, screen_time,
                            physical_activity, mood_score, sleep_debt, lifestyle_encoded]])
    predictions = model.predict(input_data)[0]
    st.session_state.predictions = predictions

# Retrieve predictions (even after rerun)
if st.session_state.predictions is not None:
    burnout, stress_level, stress_per_hr, well_score, sleep_quality, mental_fatigue = st.session_state.predictions
else:
    burnout = stress_level = stress_per_hr = well_score = sleep_quality = mental_fatigue = None

if st.session_state.predictions is not None:

# Output
    with st.container():
        st.markdown("""<div class="card"><h3>📤 Predictions</h3></div>""", unsafe_allow_html=True)
        if burnout is not None:
            display_card("Burnout Risk Score", burnout)
            display_card("Stress Level", stress_level)
            display_card("Stress per Study Hour", stress_per_hr)
            display_card("Overall Wellbeing Score", well_score)
            display_card("Sleep Quality Score", sleep_quality)
            display_card("Mental Fatigue Score", mental_fatigue)

if burnout is not None:
    st.markdown("""<div class="card"><h3>🌐 Compare Your Lifestyle</h3></div>""", unsafe_allow_html=True)

    compare_df = pd.read_csv("./data/comparisons.csv")
    groups = compare_df['group'].unique()
    selected_group = st.selectbox("Choose a lifestyle group to compare with:", groups)

    # Fetch group data
    group_data = compare_df[compare_df['group'] == selected_group].iloc[0, 1:].values.reshape(1, -1)
    group_prediction = model.predict(group_data)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧍 Your Burnout Score")
        display_card("Burnout", burnout)
        display_card("Stress Level", stress_level)

    with col2:
        st.markdown(f"#### 👥 {selected_group}")
        display_card("Burnout", group_prediction[0])
        display_card("Stress Level", group_prediction[1])
       
  # Tabs for visual insights
if st.session_state.predictions is not None:
# Only show tabs after prediction
    tab1, tab2 = st.tabs(["🕸️ Lifestyle Radar", "📈 Trendline vs Peers"])

    # --- Tab 1: Radar Chart ---
    with tab1:
        st.markdown("""<div class="card"><h4>🕸️ Lifestyle Score Overview</h4></div>""", unsafe_allow_html=True)

        categories = ['Burnout', 'Stress', 'Stress/Hour', 'Wellbeing', 'Sleep Quality', 'Mental Fatigue']
        values = [burnout, stress_level, stress_per_hr, well_score, sleep_quality, mental_fatigue]

        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Scores',
            line=dict(color='royalblue')
        ))

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=400
        )

        st.plotly_chart(radar_fig, use_container_width=True)

    # --- Tab 2: Trendline vs Peers ---
    with tab2:
        st.markdown("""<div class="card"><h4>📈 Trendline vs Peers</h4></div>""", unsafe_allow_html=True)

        peer_df = pd.read_csv("./data/comparisons.csv")
        national_avg = peer_df[peer_df["group"] == "National Average"].iloc[0, 1:].values
        selected_peer = peer_df[peer_df["group"] == selected_group].iloc[0, 1:].values
        user_scores = np.array(st.session_state.predictions[:6])

        score_labels = ["Burnout", "Stress Level", "Stress/Hour", "Wellbeing", "Sleep Quality", "Mental Fatigue"]

        df = pd.DataFrame({
            "Score Type": score_labels * 3,
            "Value": np.concatenate([user_scores, selected_peer[:6], national_avg[:6]]),
            "Source": ["You"] * 6 + [selected_group] * 6 + ["National Average"] * 6
        })

        trend_fig = px.line(df, x="Score Type", y="Value", color="Source", markers=True)
        trend_fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#D2ECFF"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(trend_fig, use_container_width=True)

if st.session_state.predictions is not None:

# Suggestions
    with st.container():
        st.markdown("""<div class="card"><h3>📌 Suggestions Based on Your Scores</h3></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if burnout is not None:
            display_suggestion_card("🔥 Burnout Risk Score", burnout)
            display_suggestion_card("💢 Stress Level", stress_level)
            display_suggestion_card("📚 Stress/Study Hour", stress_per_hr)
    
    with col2:
        if burnout is not None:
            display_suggestion_card("🧘 Overall Wellbeing Score", well_score)
            display_suggestion_card("😴 Sleep Quality Score", sleep_quality)
            display_suggestion_card("🧠 Mental Fatigue Score", mental_fatigue)
    
    # Explainable AI (SHAP + Plotly)
    if burnout is not None:
        st.markdown("""<div class="card"><h3>🔍 Why Burnout Score? (Explainable AI)</h3></div>""", unsafe_allow_html=True)
    
        # Define feature names (must match your model training order)
        features = [
        "Age", "Sleep_Hours", "Study_Hours", "Screen_Time_Post_10PM",
        "Physical_Activity_Minutes", "Mood_Score", "Sleep_Debt", "Lifestyle_Encoded"
        ]
    
    # Form the same input vector used for prediction
        user_vector = np.array([[age, sleep_hours, study_hours, screen_time,
                             physical_activity, mood_score, sleep_debt, lifestyle_encoded]])
        # Select only burnout model (index 0)
        burnout_model = model.estimators_[0]
    
        # SHAP Explainer
        explainer = shap.TreeExplainer(burnout_model)
    
        shap_values = explainer(user_vector)
    
        shap_vals = shap_values.values[0]
        percent_contributions = np.abs(shap_vals) / np.sum(np.abs(shap_vals)) * 100
        sorted_idx = np.argsort(percent_contributions)[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_percents = [percent_contributions[i] for i in sorted_idx]
        sorted_values = [shap_vals[i] for i in sorted_idx]
    
        # Plotly Bar Chart in Card
        with st.container():
            tip_map = {
        "Age": "🎯 Age may influence recovery rate and energy resilience.",
        "Sleep_Hours": "🛌 Increasing sleep hours can reduce mental fatigue and improve overall wellness.",
        "Study_Hours": "📚 High study hours can increase stress—balance with breaks.",
        "Screen_Time_Post_10PM": "📱 Reducing late-night screen time improves sleep quality and reduces stress.",
        "Physical_Activity_Minutes": "💪 Regular activity boosts endorphins and reduces burnout risk.",
        "Mood_Score": "😊 A higher mood score often reflects emotional stability and mental resilience.",
        "Sleep_Debt": "😴 Reducing sleep debt can improve focus, mood, and reduce burnout.",
        "Lifestyle_Encoded": "🧬 Different lifestyle types can influence your physical and mental energy patterns."
    }
            fig = go.Figure(data=[
                  go.Bar(
                    x=sorted_percents,
                    y=sorted_features,
                    orientation='h',
                    text=[f"{v:+.2f}" for v in sorted_values],
                    textposition='outside',
                        textfont=dict( color='#0073ff', size=14,family='Arial'),
                    hovertext=[
                        f"{sorted_features[i]} → {tip_map.get(sorted_features[i], '')}"
                        for i in range(len(sorted_features))
                    ],
        hoverinfo='text',
        marker=dict(color='rgba(255,140,0,0.7)')
    )
    
            ])
    
            fig.update_layout(
                xaxis_title="Impact on Burnout Score (%)",
                yaxis_title="Feature",
                height=400,
                margin=dict(l=60, r=30, t=30, b=30),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#B1DDF1")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
                        
        <div style="margin-top: 15px; background-color: #111; padding: 15px; border-radius: 8px;">
        <p style="color:#CCCCCC; font-size: 14px;">
            ✅ <strong>Positive (+)</strong> values mean the feature helped <span style="color:#90ee90;">reduce burnout</span>.<br>
            ❌ <strong>Negative (−)</strong> values mean the feature <span style="color:#ff726f;">increased burnout</span>.<br>
            📊 The number (like +0.35 or -0.16) shows how much it pushed your score.
        </p>
        </div>
    """, unsafe_allow_html=True)
            
# Load recommender model
@st.cache_resource
def load_recommender():
    return joblib.load("./Models/smart_recommender.pkl")

recommender_model = load_recommender()

# Generate suggestions
if st.session_state.predictions is not None:
    st.markdown("""<div class="card"><h3>🎬 Smart Lifestyle Suggestions for You</h3></div>""", unsafe_allow_html=True)

    user_input = {
        "age": age,
        "sleep": sleep_hours,
        "study": study_hours,
        "screen": screen_time,
        "activity": physical_activity,
        "mood": mood_score,
        "debt": sleep_debt,
        "lifestyle": lifestyle_encoded
    }

    suggestions = generate_suggestions(recommender_model, user_input)

    if suggestions:
        # Show first 9 in a 3x3 grid
        for row in range(3):
            cols = st.columns(3)
            for col_idx in range(3):
                idx = row * 3 + col_idx
                if idx < len(suggestions):
                    s = suggestions[idx]
                    card_html = display_resource_card(s["title"], s["url"], s["icon"])
                    cols[col_idx].markdown(card_html, unsafe_allow_html=True)


            
# Place this import at the top of your file if not already
import joblib

# 🔁 Cached model loader
@st.cache_resource
def load_forecast_model():
    return joblib.load("./Models/days_trainer.pkl")

forecast_model = load_forecast_model()

if st.session_state.predictions is not None:
    st.markdown("""
        <div class="card">
            <h3>📈 Adaptive Burnout & Wellbeing Forecast</h3>
            <p style='font-size: 14px; margin-bottom:0;'>Projected scores if your current lifestyle continues.</p>
        </div>
    """, unsafe_allow_html=True)

    # 📥 Input rows for each day
    days = [3, 7, 14, 21, 30, 45, 90, 182, 365]
    forecast_df = pd.DataFrame([{
        "Age": age,
        "Sleep_Hours": sleep_hours,
        "Study_Hours": study_hours,
        "Screen_Time_Post_10PM": screen_time,
        "Physical_Activity_Minutes": physical_activity,
        "Mood_Score": mood_score,
        "Sleep_Debt": sleep_debt,
        "Day": d,
        "Lifestyle_Score": lifestyle_encoded
    } for d in days])

    # 🛑 Check for NaNs
    if forecast_df.isnull().any().any():
        st.error("🚫 Forecast data has missing values.")
        st.dataframe(forecast_df)
        st.stop()

    # 🔮 Predict
    try:
        predictions = forecast_model.predict(forecast_df)
    except Exception as e:
        st.error(f"⚠️ Forecast model failed: {str(e)}")
        st.stop()

    forecast_plot_df = pd.DataFrame(predictions, columns=[
        "Burnout_Risk_Score", "Stress_Level", "Stress_Per_Study_Hour",
        "Overall_Wellbeing_Score", "Sleep_Quality_Score", "Mental_Fatigue_Score"
    ])
    forecast_plot_df["Day"] = days

    # 📊 Metric selection
    visible_metrics = st.multiselect(
        "📊 Select metrics to include in forecast:",
        forecast_plot_df.columns[:-1].tolist(),
        default=forecast_plot_df.columns[:-1].tolist()
    )

    # 🎞️ Animate toggle
    animate = st.checkbox("🎞️ Animate Forecast", value=True)

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    if animate:
        for i in range(1, len(forecast_plot_df) + 1):
            temp_df = forecast_plot_df.iloc[:i]
            long_df = temp_df.melt(id_vars="Day", var_name="Metric", value_name="Score")
            long_df = long_df[long_df["Metric"].isin(visible_metrics)]

            chart = alt.Chart(long_df).mark_line(point=True).encode(
                x=alt.X("Day:O", title="Day"),
                y=alt.Y("Score:Q", title="Score"),
                color="Metric:N",
                tooltip=["Day", "Metric", "Score"]
            ).properties(height=400)

            chart_placeholder.altair_chart(chart, use_container_width=True)

            current_day = int(temp_df["Day"].iloc[-1])
            current_burnout = float(temp_df["Burnout_Risk_Score"].iloc[-1])

            if current_burnout >= 85:
                status_placeholder.markdown(f"🛑 **Day {current_day}**: Burnout extremely high (**{current_burnout:.1f}**)! ⚠️")
            elif current_burnout >= 70:
                status_placeholder.markdown(f"⚠️ **Day {current_day}**: Burnout rising (**{current_burnout:.1f}**) — consider changes.")
            elif current_burnout >= 50:
                status_placeholder.markdown(f"🔄 **Day {current_day}**: Moderate burnout (**{current_burnout:.1f}**) — monitor balance.")
            else:
                status_placeholder.markdown(f"✅ **Day {current_day}**: Burnout healthy (**{current_burnout:.1f}**) — keep it up!")

            time.sleep(0.45)
    else:
        long_df = forecast_plot_df.melt(id_vars="Day", var_name="Metric", value_name="Score")
        long_df = long_df[long_df["Metric"].isin(visible_metrics)]

        chart = alt.Chart(long_df).mark_line(point=True).encode(
            x=alt.X("Day:O", title="Day"),
            y=alt.Y("Score:Q", title="Score"),
            color="Metric:N",
            tooltip=["Day", "Metric", "Score"]
        ).properties(height=400)

        chart_placeholder.altair_chart(chart, use_container_width=True)

 # 📈 Analyze burnout across all forecasted days
    burnout_scores = forecast_plot_df["Burnout_Risk_Score"].tolist()
    days = forecast_plot_df["Day"].tolist()
    
    min_score = min(burnout_scores)
    max_score = max(burnout_scores)
    start_score = burnout_scores[0]
    end_score = burnout_scores[-1]
    
    # Calculate net change
    delta = end_score - start_score
    
    # Get trend direction (very basic linearity check)
    rising = all(earlier <= later for earlier, later in zip(burnout_scores, burnout_scores[1:]))
    falling = all(earlier >= later for earlier, later in zip(burnout_scores, burnout_scores[1:]))
    
    if rising and delta > 5:
        st.error(f"🛑 Burnout is **steadily increasing** from {start_score:.1f} to {end_score:.1f} over the year — serious lifestyle adjustments are needed.")
    elif falling and delta < -5:
        st.success(f"✅ Burnout is **consistently decreasing** from {start_score:.1f} to {end_score:.1f} — great improvement ahead!")
    elif max_score - min_score < 8:
        st.info(f"📊 Burnout remains **relatively stable** between {min_score:.1f} and {max_score:.1f} throughout the year.")
    else:
        st.warning(f"⚠️ Burnout shows **fluctuating trend** (from {min_score:.1f} to {max_score:.1f}). Watch out for possible rebounds — try to maintain healthy consistency.")
    
    
    
    
     
# About Section — Always Visible with Anchor
with st.container():
    st.markdown('<div id="about"></div>', unsafe_allow_html=True)
    display_about_card()

with st.container():
    st.markdown('<div id="story"></div>', unsafe_allow_html=True)
    display_story_card()

st.markdown('<a id="top-btn" href="#top">🔝 Top</a>', unsafe_allow_html=True)


