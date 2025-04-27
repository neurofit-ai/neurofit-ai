import time
import random
import streamlit as st
import joblib
import pandas as pd
import account_details  # Ensure this is used for profile

data = pd.read_csv("athlete_performance_large.csv")

# ==================== CONSTANTS & INITIALIZATION ====================
CATEGORICAL_COLS = ['Gender', 'Sport_Type', 'Exercise_Type', 'Intensity_Level', 'Hydration_Level']
MODEL_FILES = {
    'endurance': 'models/endurance_model.pkl',
    'injury': 'models/injury_model.pkl',
    'calories': 'models/calories_model.pkl',
    'sleep': 'models/sleep_model.pkl',
    'encoders': 'models/label_encoders.pkl',
    'protein': 'models/protein_model.pkl',
    'carbs': 'models/carbs_model.pkl',
    'fats': 'models/fats_model.pkl'
}

# ==================== RECOMMENDATION SYSTEMS ====================
def generate_diet_plan(predictions, user_input):
    calorie_needs = predictions['calories']
    endurance = predictions['endurance_score']
    weight = user_input.get('Weight', 70)

    protein_grams = predictions['Protein_g']
    carb_grams = predictions['Carbs_g']
    fat_grams = predictions['Fats_g']

    diet_plan = {
        'Morning': [
            f"🍳 Protein: {round(protein_grams*0.3)}g (Eggs/Greek Yogurt)",
            f"🌾 Carbs: {round(carb_grams*0.3)}g (Oatmeal/Whole Grain)",
            "🥑 Healthy Fats: 1/2 Avocado",
            "💧 Hydration: 500ml water + electrolytes"
        ],
        'Pre-Workout': [
            f"🍌 Carbs: {round(carb_grams*0.2)}g (Banana + Toast)",
            "☕ Caffeine: Optional (100-200mg)"
        ],
        'Post-Workout': [
            f"🍗 Protein: {round(protein_grams*0.4)}g (Chicken/Fish/Plant Protein)",
            f"🍠 Carbs: {round(carb_grams*0.4)}g (Sweet Potato/Rice)",
            "🥜 Fats: 1 tbsp Nut Butter",
            "💧 Hydration: 500ml water + electrolytes"
        ],
        'Evening': [
            f"🥩 Protein: {round(protein_grams*0.3)}g (Lean Meat/Tofu)",
            f"🥦 Carbs: {round(carb_grams*0.1)}g (Vegetables)",
            "🥜 Fats: 1 oz Nuts/Seeds",
            "💧 Hydration: 500ml water"
        ],
        'Snacks': [
            "Greek Yogurt with Berries",
            "Handful of Nuts",
            "Protein Shake (if needed)"
        ]
    }

    if predictions['sleep_hours'] > 8:
        diet_plan['Evening'].append("🍵 Herbal Tea (Chamomile/Lavender)")
    if endurance > 7:
        diet_plan['Post-Workout'].append("+100g Complex Carbs for Recovery")

    return diet_plan

def generate_mental_advice(predictions, user_input):
    injury_risk = predictions['injury_risk']
    fatigue = user_input.get('Fatigue_Score', 5)
    sleep = predictions['sleep_hours']

    advice = []

    if fatigue > 7:
        advice.append("🧘 High Fatigue: Try 2x daily 10-min meditation sessions")
        advice.append("🌿 Adaptogens: Consider ashwagandha or rhodiola supplements")
    elif fatigue >= 5:
        advice.append("😌 Moderate Fatigue: Practice deep breathing exercises 3x/day")

    if injury_risk == 'Severe':
        advice.append("🩹 Critical Recovery Needed: Schedule sports massage and reduce intensity by 50% this week")
    elif injury_risk == 'Moderate':
        advice.append("⚠️ Injury Warning: Increase warm-up time to 20 minutes")

    if sleep < 6:
        advice.append("💤 Sleep Deficiency: Try 1-3mg melatonin 1hr before bed")
    advice.append(f"⏰ Sleep Consistency: Maintain {sleep:.1f} hour sleep schedule")
    advice.append("📝 Daily Reflection: Spend 5 minutes journaling post-workout")
    advice.append("🌳 Nature Therapy: Add 20-min outdoor walks 3x/week")

    return advice

# ==================== PDF GENERATION ====================
# ==================== PDF GENERATION ====================
def generate_pdf_report(predictions, diet, advice, user_input):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
    
    # Create a buffer to store PDF
    import io
    buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='CenterTitle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        spaceAfter=14
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        spaceBefore=12,
        spaceAfter=6
    ))
    
    content = []
    
    # Header
    content.append(Paragraph('NeuroFit', styles['CenterTitle']))
    content.append(Paragraph('Athlete Performance Report', styles['CenterTitle']))
    
    # Performance Metrics
    content.append(Paragraph('Performance Metrics', styles['SectionTitle']))
    metrics = f"""<b>Endurance Score:</b> {predictions['endurance_score']}/10<br/>
<b>Calories Needed:</b> {predictions['calories']} kcal<br/>
<b>Injury Risk:</b> {predictions['injury_risk']}<br/>
<b>Optimal Sleep:</b> {predictions['sleep_hours']} hours<br/>
<b>Protein Needs:</b> {predictions['Protein_g']} g<br/>
<b>Carbohydrates Needs:</b> {predictions['Carbs_g']} g<br/>
<b>Fats Needs:</b> {predictions['Fats_g']} g"""
    content.append(Paragraph(metrics, styles['Normal']))
    content.append(Spacer(1, 12))
    
    # Personalized Nutrition Plan
    content.append(Paragraph('Personalized Nutrition Plan', styles['SectionTitle']))
    diet_text = []
    for meal, items in diet.items():
        meal_items = []
        for item in items:
            clean_item = ''.join(char for char in item if ord(char) < 256)
            meal_items.append(f"• {clean_item}")
        diet_text.append(f"<b>{meal}:</b><br/>" + "<br/>".join(meal_items))
    content.append(Paragraph("<br/><br/>".join(diet_text), styles['Normal']))
    content.append(Spacer(1, 12))
    
    # Mental Performance Guide
    content.append(Paragraph('Mental Performance Guide', styles['SectionTitle']))
    clean_advice = [''.join(char for char in item if ord(char) < 256) for item in advice]
    advice_text = "• " + "<br/>• ".join(clean_advice)
    content.append(Paragraph(advice_text, styles['Normal']))
    content.append(Spacer(1, 12))
    
    # User Details
    content.append(Paragraph('User Details', styles['SectionTitle']))
    details = f"""<b>Age:</b> {user_input['Age']}<br/>
<b>Gender:</b> {user_input['Gender']}<br/>
<b>Weight:</b> {user_input['Weight']} kg<br/>
<b>Sport:</b> {user_input['Sport_Type']}<br/>
<b>Exercise Type:</b> {user_input['Exercise_Type']}"""
    content.append(Paragraph(details, styles['Normal']))
    
    # Build PDF
    doc.build(content)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

# ==================== PREDICTION ENGINE ====================
def load_models():
    models = {}
    try:
        models['endurance'] = joblib.load(MODEL_FILES['endurance'])
        models['injury'] = joblib.load(MODEL_FILES['injury'])
        models['calories'] = joblib.load(MODEL_FILES['calories'])
        models['sleep'] = joblib.load(MODEL_FILES['sleep'])
        models['protein'] = joblib.load(MODEL_FILES['protein'])
        models['carbs'] = joblib.load(MODEL_FILES['carbs'])
        models['fats'] = joblib.load(MODEL_FILES['fats'])
        models['encoders'] = joblib.load(MODEL_FILES['encoders'])
        return models
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def predict_performance(models, user_input):
    try:
        input_df = pd.DataFrame([user_input])
        for col in CATEGORICAL_COLS:
            if col in input_df.columns:
                try:
                    le = models['encoders'][col]
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    input_df[col] = 0

        endurance = max(0, min(10, models['endurance'].predict(input_df)[0]))
        calories = max(1500, min(5000, models['calories'].predict(input_df)[0]))
        sleep = max(4, min(12, models['sleep'].predict(input_df)[0]))
        protein = max(50, min(300, models['protein'].predict(input_df)[0]))
        carbs = max(100, min(800, models['carbs'].predict(input_df)[0]))
        fats = max(20, min(150, models['fats'].predict(input_df)[0]))

        # Get injury probability and classify using thresholds
        injury_proba = models['injury'].predict_proba(input_df)[0]
        max_prob = max(injury_proba)  # Get highest probability
        injury_risk = (
            "Minor" if 0.25 <= max_prob < 0.5 else
            "Moderate" if 0.5 <= max_prob < 0.75 else
            "Severe"
        )

        return {
            'endurance_score': round(endurance, 2),
            'calories': round(calories),
            'sleep_hours': round(sleep, 1),
            'injury_risk': injury_risk,
            'raw_injury_prob': round(max_prob * 100, 2),
            'Protein_g': round(protein),
            'Carbs_g': round(carbs),
            'Fats_g': round(fats)
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# ==================== STREAMLIT INTERFACE ====================

def logout():
    """Reset session states and rerun to show login page"""
    st.session_state.authenticated = False
    st.session_state.show_signup = False
    if 'user_email' in st.session_state:
        del st.session_state.user_email
    st.rerun()

def profile():
    st.session_state.show_account = True
    st.rerun()

def calculate_metric_scores(predictions):
    endurance_score = predictions['endurance_score'] * 10
    injury_score = predictions['raw_injury_prob']  # Direct probability percentage
    sleep_score = min(predictions['sleep_hours'] / 10 * 100, 100)
    
    return {
        'Endurance': endurance_score,
        'Injury Risk': injury_score,
        'Sleep Quality': sleep_score
    }

def dashboard():
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'show_account' not in st.session_state:
        st.session_state.show_account = False
    if 'login_time' not in st.session_state:
        st.session_state.login_time = time.time()
    if 'models_loaded' not in st.session_state:
        st.session_state.models = load_models()
        st.session_state.models_loaded = st.session_state.models is not None
    if 'user_details' not in st.session_state:
        st.session_state.user_details = {'Age': 25}
    
    """Null-safe dashboard"""
    if not st.session_state.get("user_details"):
        st.error("Session expired. Re-login required.")
        logout()
        return

    if st.session_state.get("show_account", False):
        account_details.account()
        return

    col1, col2 = st.columns([5, 2])
    with col2:
        profile_menu = st.selectbox(
            label="Account Details",
            options=["Profile", "Logout"],
            index=None,
            placeholder="👤 Account Details",
            label_visibility="collapsed"
        )
        if profile_menu == "Profile":
            profile()
        elif profile_menu == "Logout":
            with st.spinner("Logging out..."):
                time.sleep(2)
                logout()

    st.markdown("## 🏋️‍♂️ Athlete Information")
    col1, col2, col3 = st.columns([1, 1.5, 1.5])


    age = st.session_state.user_details['Age']
    gender = st.session_state.user_details['Gender']

    with col1:
        sport = st.selectbox("Primary Sport", data.Sport_Type.unique().tolist(), index=None, placeholder="Enter your primary sport")
        height = st.number_input("Height (cm)", min_value=150, max_value=220, value=None, placeholder="Enter your height")
        hydration = st.selectbox("Hydration Level", ["Low", "Moderate", "High"], index=None, placeholder="Enter your hydration level")

    with col2:
        exercise_type = st.selectbox("Exercise Type", ["Endurance", "Strength", "HIIT", "Skill Training", "Recovery"], index=None, placeholder="Enter your exercise type")
        weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=None, placeholder="Enter your weight")
        speed_score = st.number_input("Running Speed (km/h)", min_value=10.0, max_value=30.0, value=None, placeholder="Enter your speed score", step=0.01)

    with col3:
        duration = st.number_input("Duration (min)", min_value=10, max_value=300, value=None, placeholder="Enter your workout duration")
        intensity = st.selectbox("Intensity Level", ["Low", "Medium", "High"], index=None, placeholder="Enter your intensity level")    
        fatigue = st.slider("Fatigue (1-10)", 1, 10, 8)
    
    def check_details():
        if age is None or exercise_type is None or height is None or hydration is None or gender is None or duration is None or weight is None or speed_score is None or sport is None or intensity is None or fatigue is None:
            st.error("Please fill in all fields.")
            return False
        return True

    st.html("<hr>")

    if st.button("Analyze My Performance", type="primary", use_container_width=True):
        if not st.session_state.models_loaded:
            st.error("Models failed to load. Please check model files.")
            return

        try:
            heart_rate = 220 - age
            strength_score = 100 / weight if weight < 100 else 150 / weight

            with st.spinner("Analyzing your performance..."):
                user_input = {
                    'Age': age,
                    'Gender': gender,
                    'Weight': weight,
                    'Sport_Type': sport,
                    'Exercise_Type': exercise_type,
                    'Duration_Minutes': duration,
                    'Intensity_Level': intensity,
                    'Hydration_Level': hydration,
                    'Fatigue_Score': fatigue,
                    'Heart_Rate': heart_rate,
                    'Speed_Score': speed_score,
                    'Strength_Score': strength_score
                }

                st.session_state.predictions = predict_performance(st.session_state.models, user_input)
        except TypeError:
            check_details()

    if st.session_state.predictions:
        st.markdown("## 📝 Performance Summary")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Key Metrics")
            st.metric("Endurance Score", f"{st.session_state.predictions['endurance_score']}/10")
            st.metric("Injury Risk", st.session_state.predictions['injury_risk'])
        with col2:
            st.markdown("### Key Recommendations")
            st.write("- Follow recommended nutrition plan from PDF report")
            st.write("- Implement recovery strategies based on injury risk")
            st.write("- Monitor sleep patterns and adjust training accordingly")
        
        st.html("<hr>")

        with st.spinner("Visualizing your performance..."):
            st.markdown("## Performance Report")
            metric_scores = calculate_metric_scores(st.session_state.predictions)
            df_metrics = pd.DataFrame({
                'Metrics': list(metric_scores.keys()),
                'Values': list(metric_scores.values())
            })

        df_metrics_pivot = df_metrics.pivot(
            columns='Metrics',
            values='Values'
        ).reset_index(drop=True)

        bar_colors = {
            "Endurance": "#4B9BFF",
            "Injury Risk": "#FF4B4B",
            "Sleep Quality": "#2CA02C"
        }

        st.bar_chart(
            df_metrics_pivot,
            color=[bar_colors[col] for col in df_metrics_pivot.columns],
            height=400
        )

        st.caption("🔹 Higher scores are better for Endurance and Sleep. Lower is better for Injury Risk.")
        
        st.html("<hr>")

        st.markdown("## 🥗 Nutrition Recommendations (Needed)")
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        with col1:
            st.metric("Proteins", f"{st.session_state.predictions['Protein_g']} g")
        with col2:
            st.metric("Carbohydrates", f"{st.session_state.predictions['Carbs_g']} g") 
        with col3:
            st.metric("Fats", f"{st.session_state.predictions['Fats_g']} g")
        with col4:
            sleep_hours = st.session_state.predictions['sleep_hours']
            st.metric("Sleep Hours", f"{sleep_hours} hrs")
        with col5:
            calories = st.session_state.predictions['calories']
            st.metric("Calories", f"{calories} kcal")

        diet = generate_diet_plan(st.session_state.predictions, {'Weight': weight})
        advice = generate_mental_advice(st.session_state.predictions, {'Fatigue_Score': fatigue})
        pdf_bytes = generate_pdf_report(st.session_state.predictions, diet, advice, {
            'Age': age,
            'Gender': gender,
            'Weight': weight,
            'Sport_Type': sport,
            'Exercise_Type': exercise_type
        })

        st.download_button("📥 Download Full Report", data=pdf_bytes, file_name="athlete_performance_report.pdf", mime="application/pdf", use_container_width=True)

if __name__ == "__main__":
    dashboard()
