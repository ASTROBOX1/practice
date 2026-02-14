import streamlit as st
import pandas as pd
import joblib

# 1. تحميل الـ Pipeline الذي قمت بحفظه
model = joblib.load("my_pipeline.joblib")

# إعدادات واجهة التطبيق
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🩺 تطبيق التنبؤ بأمراض القلب")
st.write("أدخل بيانات المريض أدناه للحصول على التنبؤ:")

# 2. إنشاء نموذج المدخلات (Form) بناءً على أعمدة ملفك
with st.form("patient_data"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("العمر (Age)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("الجنس (Sex)", options=["M", "F"])
        chest_pain = st.selectbox("نوع ألم الصدر (ChestPainType)", options=['ATA', 'NAP', 'ASY', 'TA'])
        resting_bp = st.number_input("ضغط الدم في الراحة (RestingBP)", value=120)
        cholesterol = st.number_input("الكوليسترول (Cholesterol)", value=200)
        
    with col2:
        fasting_bs = st.selectbox("سكر الدم صائم > 120 (FastingBS)", options=[0, 1])
        resting_ecg = st.selectbox("تخطيط القلب (RestingECG)", options=['Normal', 'ST', 'LVH'])
        max_hr = st.number_input("أقصى معدل ضربات قلب (MaxHR)", value=150)
        exercise_angina = st.selectbox("ذبحة صدرية ناتجة عن التمرين (ExerciseAngina)", options=["N", "Y"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", value=0.0, format="%.1f")
        st_slope = st.selectbox("ميل قطاع ST (ST_Slope)", options=['Up', 'Flat', 'Down'])

    submit = st.form_submit_button("تحليل الحالة")

# 3. معالجة البيانات عند الضغط على الزر
if submit:
    # إنشاء DataFrame بنفس أسماء الأعمدة الأصلية في ملفك
    input_data = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }])

    # التنبؤ باستخدام الـ Pipeline (سيتولى الـ Encoding و الـ Scaling آلياً)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # 4. عرض النتائج
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ احتمالية عالية للإصابة بمرض القلب ({probability:.2%})")
    else:
        st.success(f"✅ الاحتمالية منخفضة، الحالة تبدو مستقرة ({probability:.2%})")