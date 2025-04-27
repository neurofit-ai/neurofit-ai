import time
import random
import streamlit as st
import joblib
import pandas as pd
from fpdf import FPDF
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
def generate_pdf_report(predictions, diet, advice, user_input):
    """
    Generate a PDF performance report with error handling at every stage.
    Returns PDF bytes if successful, None otherwise.
    """
    try:
        # Validate inputs
        if not all([predictions, diet, advice, user_input]):
            raise ValueError("Missing required input data")
            
        if not isinstance(predictions, dict):
            raise TypeError("Predictions must be a dictionary")
            
        # Initialize PDF
        class PDF(FPDF):
            def header(self):
                try:
                    self.set_font('Arial', 'B', 16)
                    self.cell(0, 10, 'NeuroFit', 0, 1, 'C')
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 10, 'Athlete Performance Report', 0, 1, 'C')
                    self.ln(5)
                except Exception as e:
                    raise RuntimeError(f"PDF header failed: {str(e)}")

            def chapter_title(self, title):
                try:
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 10, str(title), 0, 1)
                    self.ln(2)
                except Exception as e:
                    raise RuntimeError(f"PDF title failed: {str(e)}")

            def chapter_body(self, body):
                try:
                    self.set_font('Arial', '', 12)
                    self.multi_cell(0, 7, str(body))
                    self.ln()
                except Exception as e:
                    raise RuntimeError(f"PDF body failed: {str(e)}")

        pdf = PDF()
        pdf.add_page()

        # 1. Performance Metrics
        try:
            metrics_data = [
                f"Endurance Score: {predictions.get('endurance_score', 'N/A')}/10",
                f"Calories Needed: {predictions.get('calories', 'N/A')} kcal",
                f"Injury Risk: {predictions.get('injury_risk', 'N/A')}",
                f"Optimal Sleep: {predictions.get('sleep_hours', 'N/A')} hours",
                f"Protein Needs: {predictions.get('Protein_g', 'N/A')} g",
                f"Carbohydrates Needs: {predictions.get('Carbs_g', 'N/A')} g",
                f"Fats Needs: {predictions.get('Fats_g', 'N/A')} g"
            ]
            pdf.chapter_title('Performance Metrics')
            pdf.chapter_body("\n".join(metrics_data))
        except Exception as e:
            raise RuntimeError(f"Metrics section failed: {str(e)}")

        # 2. Nutrition Plan
        try:
            diet_text = []
            for meal, items in diet.items():
                try:
                    meal_items = []
                    for item in items:
                        clean_item = ''.join(char for char in str(item) if ord(char) < 256)
                        meal_items.append(clean_item)
                    diet_text.append(f"{meal}:\n" + "\n".join(meal_items))
                except Exception as e:
                    raise RuntimeError(f"Diet item processing failed: {str(e)}")
            
            pdf.chapter_title('Personalized Nutrition Plan')
            pdf.chapter_body("\n".join(diet_text))
        except Exception as e:
            raise RuntimeError(f"Nutrition section failed: {str(e)}")

        # 3. Mental Performance Guide
        try:
            clean_advice = []
            for item in advice:
                try:
                    clean_advice.append(''.join(char for char in str(item) if ord(char) < 256))
                except:
                    clean_advice.append("[Advice item unavailable]")
            
            pdf.chapter_title('Mental Performance Guide')
            pdf.chapter_body("\n".join(clean_advice))
        except Exception as e:
            raise RuntimeError(f"Advice section failed: {str(e)}")

        # 4. User Details
        try:
            user_details = [
                f"Age: {user_input.get('Age', 'N/A')}",
                f"Gender: {user_input.get('Gender', 'N/A')}",
                f"Weight: {user_input.get('Weight', 'N/A')} kg",
                f"Sport: {user_input.get('Sport_Type', 'N/A')}",
                f"Exercise Type: {user_input.get('Exercise_Type', 'N/A')}"
            ]
            pdf.chapter_title('User Details')
            pdf.chapter_body("\n".join(user_details))
        except Exception as e:
            raise RuntimeError(f"User details failed: {str(e)}")

        # Finalize PDF
        try:
            return pdf.output(dest='S').encode('latin-1', 'replace')
        except Exception as e:
            raise RuntimeError(f"PDF finalization failed: {str(e)}")

    except Exception as e:
        import traceback
        traceback.print_exc()
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

    required_keys = ['predictions', 'user_details']
    if not all(k in st.session_state for k in required_keys):
        st.error("Missing required data. Please complete the analysis first.")
        return

    # Generate performance metrics display
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Key Metrics")
        st.metric("Endurance Score", 
                f"{st.session_state.predictions.get('endurance_score', 'N/A')}/10")
        st.metric("Injury Risk", 
                st.session_state.predictions.get('injury_risk', 'N/A'))
    
    with col2:
        st.markdown("### Key Recommendations")
        st.write("- Follow recommended nutrition plan")
        st.write("- Implement recovery strategies")
        st.write("- Monitor sleep patterns")

    # Generate and display nutrition metrics
    st.markdown("## 🥗 Nutrition Recommendations")
    cols = st.columns(5)
    with cols[0]:
        st.metric("Proteins", f"{st.session_state.predictions.get('Protein_g', 'N/A')} g")
    with cols[1]:
        st.metric("Carbs", f"{st.session_state.predictions.get('Carbs_g', 'N/A')} g")
    with cols[2]:
        st.metric("Fats", f"{st.session_state.predictions.get('Fats_g', 'N/A')} g")
    with cols[3]:
        st.metric("Sleep", f"{st.session_state.predictions.get('sleep_hours', 'N/A')} hrs")
    with cols[4]:
        st.metric("Calories", f"{st.session_state.predictions.get('calories', 'N/A')} kcal")

    # Generate report data
    try:
        diet_plan = generate_diet_plan(
            st.session_state.predictions,
            {'Weight': st.session_state.user_details.get('Weight', 70)}
        )
        mental_advice = generate_mental_advice(
            st.session_state.predictions,
            {'Fatigue_Score': st.session_state.get('fatigue', 5)}
        )
    except Exception as e:
        st.error(f"Failed to generate report data: {str(e)}")
        return

    # Generate and download PDF report
    if diet_plan and mental_advice:
        pdf_bytes = generate_pdf_report(
            predictions=st.session_state.predictions,
            diet=diet_plan,
            advice=mental_advice,
            user_input={
                'Age': st.session_state.user_details.get('Age', ''),
                'Gender': st.session_state.user_details.get('Gender', ''),
                'Weight': st.session_state.user_details.get('Weight', ''),
                'Sport_Type': st.session_state.get('sport', ''),
                'Exercise_Type': st.session_state.get('exercise_type', '')
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
            st.warning("Failed to generate PDF report")
    else:
        st.error("Incomplete data for report generation")

if __name__ == "__main__":
    dashboard()
