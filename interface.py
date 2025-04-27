import time
import random
import streamlit as st
import joblib
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import account_details

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
    try:
        calorie_needs = predictions.get('calories', 2000)
        endurance = predictions.get('endurance_score', 5)
        weight = user_input.get('Weight', 70)

        protein_grams = predictions.get('Protein_g', 100)
        carb_grams = predictions.get('Carbs_g', 300)
        fat_grams = predictions.get('Fats_g', 70)

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

        if predictions.get('sleep_hours', 7) > 8:
            diet_plan['Evening'].append("🍵 Herbal Tea (Chamomile/Lavender)")
        if endurance > 7:
            diet_plan['Post-Workout'].append("+100g Complex Carbs for Recovery")

        return diet_plan
    except Exception as e:
        st.error(f"Diet plan generation failed: {str(e)}")
        return None

def generate_mental_advice(predictions, user_input):
    try:
        injury_risk = predictions.get('injury_risk', 'Moderate')
        fatigue = user_input.get('Fatigue_Score', 5)
        sleep = predictions.get('sleep_hours', 7)

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
    except Exception as e:
        st.error(f"Mental advice generation failed: {str(e)}")
        return None

# ==================== PDF GENERATION (REPORTLAB) ====================
def generate_pdf_report(predictions, diet, advice, user_input):
    try:
        buffer = BytesIO()
        
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='CenterTitle',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            spaceAfter=12,
            fontSize=16,
            textColor='#333333'
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            spaceBefore=12,
            spaceAfter=6,
            textColor='#0066cc'
        ))
        
        content = []
        
        # Header
        content.append(Paragraph("NeuroFit Performance Report", styles['CenterTitle']))
        content.append(Spacer(1, 24))
        
        # 1. Performance Metrics
        content.append(Paragraph("Performance Metrics", styles['SectionHeader']))
        metrics = [
            f"<b>Endurance Score:</b> {predictions.get('endurance_score', 'N/A')}/10",
            f"<b>Calories Needed:</b> {predictions.get('calories', 'N/A')} kcal",
            f"<b>Injury Risk:</b> {predictions.get('injury_risk', 'N/A')}",
            f"<b>Optimal Sleep:</b> {predictions.get('sleep_hours', 'N/A')} hours",
            f"<b>Protein Needs:</b> {predictions.get('Protein_g', 'N/A')} g",
            f"<b>Carbohydrates Needs:</b> {predictions.get('Carbs_g', 'N/A')} g",
            f"<b>Fats Needs:</b> {predictions.get('Fats_g', 'N/A')} g"
        ]
        for metric in metrics:
            content.append(Paragraph(metric, styles['Normal']))
            content.append(Spacer(1, 4))
        content.append(Spacer(1, 12))
        
        # 2. Nutrition Plan
        content.append(Paragraph("Personalized Nutrition Plan", styles['SectionHeader']))
        if diet:
            for meal, items in diet.items():
                content.append(Paragraph(f"<b>{meal}:</b>", styles['Normal']))
                for item in items:
                    content.append(Paragraph(f"• {item}", styles['Normal']))
                    content.append(Spacer(1, 2))
                content.append(Spacer(1, 6))
        else:
            content.append(Paragraph("No diet plan available", styles['Normal']))
        content.append(Spacer(1, 12))
        
        # 3. Mental Performance Guide
        content.append(Paragraph("Mental Performance Guide", styles['SectionHeader']))
        if advice:
            for item in advice:
                content.append(Paragraph(f"• {item}", styles['Normal']))
                content.append(Spacer(1, 4))
        else:
            content.append(Paragraph("No advice available", styles['Normal']))
        content.append(Spacer(1, 12))
        
        # 4. User Details
        content.append(Paragraph("User Details", styles['SectionHeader']))
        details = [
            f"<b>Age:</b> {user_input.get('Age', 'N/A')}",
            f"<b>Gender:</b> {user_input.get('Gender', 'N/A')}",
            f"<b>Weight:</b> {user_input.get('Weight', 'N/A')} kg",
            f"<b>Sport:</b> {user_input.get('Sport_Type', 'N/A')}",
            f"<b>Exercise Type:</b> {user_input.get('Exercise_Type', 'N/A')}"
        ]
        for detail in details:
            content.append(Paragraph(detail, styles['Normal']))
            content.append(Spacer(1, 4))
        
        doc.build(content)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
        
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None

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

        injury_proba = models['injury'].predict_proba(input_df)[0]
        max_prob = max(injury_proba)
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
    st.session_state.authenticated = False
    st.session_state.show_signup = False
    if 'user_email' in st.session_state:
        del st.session_state.user_email
    st.rerun()

def profile():
    st.session_state.show_account = True
    st.rerun()

def calculate_metric_scores(predictions):
    endurance_score = predictions.get('endurance_score', 0) * 10
    injury_score = predictions.get('raw_injury_prob', 0)
    sleep_score = min(predictions.get('sleep_hours', 0) / 10 * 100, 100)
    
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
        st.session_state.user_details = {'Age': 25, 'Gender': 'Male'}

    if not st.session_state.get("user_details"):
        st.error("Session expired. Please login again.")
        logout()
        return

    if st.session_state.get("show_account", False):
        account_details.account()
        return

    # Account controls
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
                time.sleep(1)
                logout()

    st.markdown("## 🏋️‍♂️ Athlete Information")
    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    age = st.session_state.user_details.get('Age', 25)
    gender = st.session_state.user_details.get('Gender', 'Male')

    with col1:
        sport = st.selectbox("Primary Sport", data.Sport_Type.unique().tolist(), index=None, placeholder="Enter your primary sport")
        height = st.number_input("Height (cm)", min_value=150, max_value=220, value=None, placeholder="Enter your height")
        hydration = st.selectbox("Hydration Level", ["Low", "Moderate", "High"], index=None, placeholder="Enter your hydration level")

    with col2:
        exercise_type = st.selectbox("Exercise Type", ["Endurance", "Strength", "HIIT", "Skill Training", "Recovery"], index=None, placeholder="Enter your exercise type")
        weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=None, placeholder="Enter your weight")
        speed_score = st.number_input("Running Speed (km/h)", min_value=10.0, max_value=30.0, value=None, placeholder="Enter your speed score", step=0.1)

    with col3:
        duration = st.number_input("Duration (min)", min_value=10, max_value=300, value=None, placeholder="Enter your workout duration")
        intensity = st.selectbox("Intensity Level", ["Low", "Medium", "High"], index=None, placeholder="Enter your intensity level")    
        fatigue = st.slider("Fatigue (1-10)", 1, 10, 5)

    def validate_inputs():
        required = {
            'Age': age,
            'Gender': gender,
            'Sport': sport,
            'Height': height,
            'Exercise Type': exercise_type,
            'Weight': weight,
            'Duration': duration,
            'Intensity': intensity
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
            return False
        return True

    st.markdown("---")

    if st.button("Analyze My Performance", type="primary", use_container_width=True):
        if not validate_inputs():
            return

        if not st.session_state.models_loaded:
            st.error("Models failed to load. Please try again.")
            return

        with st.spinner("Analyzing your performance..."):
            try:
                heart_rate = 220 - age
                strength_score = 100 / weight if weight < 100 else 150 / weight

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
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

    if st.session_state.predictions:
        st.markdown("## 📝 Performance Summary")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Key Metrics")
            st.metric("Endurance Score", f"{st.session_state.predictions.get('endurance_score', 'N/A')}/10")
            st.metric("Injury Risk", st.session_state.predictions.get('injury_risk', 'N/A'))
        with col2:
            st.markdown("### Key Recommendations")
            st.write("- Follow recommended nutrition plan")
            st.write("- Implement recovery strategies")
            st.write("- Monitor sleep patterns")

        st.markdown("---")

        with st.spinner("Preparing visualizations..."):
            st.markdown("## Performance Report")
            metric_scores = calculate_metric_scores(st.session_state.predictions)
            df_metrics = pd.DataFrame({
                'Metrics': list(metric_scores.keys()),
                'Values': list(metric_scores.values())
            })

            st.bar_chart(
                df_metrics.set_index('Metrics'),
                color=["#4B9BFF", "#FF4B4B", "#2CA02C"],
                height=400
            )
            st.caption("🔹 Higher scores are better for Endurance and Sleep. Lower is better for Injury Risk.")

        st.markdown("---")

        st.markdown("## 🥗 Nutrition Recommendations")
        cols = st.columns(5)
        with cols[0]:
            st.metric("Proteins", f"{st.session_state.predictions.get('Protein_g', 'N/A')}g")
        with cols[1]:
            st.metric("Carbs", f"{st.session_state.predictions.get('Carbs_g', 'N/A')}g")
        with cols[2]:
            st.metric("Fats", f"{st.session_state.predictions.get('Fats_g', 'N/A')}g")
        with cols[3]:
            st.metric("Sleep", f"{st.session_state.predictions.get('sleep_hours', 'N/A')}hrs")
        with cols[4]:
            st.metric("Calories", f"{st.session_state.predictions.get('calories', 'N/A')}kcal")

        try:
            diet = generate_diet_plan(st.session_state.predictions, {'Weight': weight})
            advice = generate_mental_advice(st.session_state.predictions, {'Fatigue_Score': fatigue})
            
            if diet and advice:
                pdf_bytes = generate_pdf_report(
                    st.session_state.predictions,
                    diet,
                    advice,
                    {
                        'Age': age,
                        'Gender': gender,
                        'Weight': weight,
                        'Sport_Type': sport,
                        'Exercise_Type': exercise_type
                    }
                )

                if pdf_bytes:
                    st.download_button(
                        "📥 Download Full Report",
                        data=pdf_bytes,
                        file_name="neurofit_performance_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.warning("Could not generate PDF report")
            else:
                st.error("Missing data for report generation")
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")

if __name__ == "__main__":
    dashboard()
