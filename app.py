import streamlit as st
import joblib
import pandas as pd
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Alzheimer's Diagnosis Pro", layout="wide")

# تحميل الموديل (تأكد من وضع اسم الملف الصحيح هنا)
@st.cache_resource
def load_alzheimer_model():
    try:
        return joblib.load('pipeline.joblib') # ضع اسم ملفك هنا
    except:
        st.error("لم يتم العثور على ملف النموذج. تأكد من وجود ملف الـ joblib في نفس المجلد.")
        return None

model = load_alzheimer_model()

st.title("🧠 نظام التنبؤ الذكي بمرض الزهايمر")
st.markdown("---")

if model:
    st.sidebar.header("معلومات عن المشروع")
    st.sidebar.info("هذا النموذج تم تدريبه باستخدام بيانات سريرية واختبارات معرفية للتنبؤ باحتمالية الإصابة بالزهايمر.")

    # تقسيم المدخلات لمجموعات منطقية
    with st.expander("📊 البيانات الديموغرافية والأساسية", expanded=True):
        col1, col2, col3 = st.columns(3)
        age = col1.slider("العمر", 60, 90, 75)
        gender = col2.selectbox("الجنس", [0, 1], format_func=lambda x: "ذكر" if x == 0 else "أنثى")
        ethnicity = col3.selectbox("العرق", [0, 1, 2, 3])
        edu = col1.selectbox("مستوى التعليم", [0, 1, 2, 3])
        bmi = col2.number_input("مؤشر كتلة الجسم (BMI)", 15.0, 40.0, 25.0)
        smoking = col3.selectbox("التدخين", [0, 1], format_func=lambda x: "غير مدخن" if x == 0 else "مدخن")

    with st.expander("🏥 التاريخ الطبي والمؤشرات الحيوية"):
        c1, c2, c3 = st.columns(3)
        alcohol = c1.slider("استهلاك الكحول", 0.0, 20.0, 5.0)
        physical = c2.slider("النشاط البدني", 0.0, 10.0, 5.0)
        diet = c3.slider("جودة النظام الغذائي", 0.0, 10.0, 5.0)
        sleep = c1.slider("جودة النوم", 0.0, 10.0, 5.0)
        family_h = c2.selectbox("تاريخ عائلي", [0, 1])
        cardio = c3.selectbox("أمراض القلب", [0, 1])
        diabetes = c1.selectbox("السكري", [0, 1])
        depress = c2.selectbox("الاكتئاب", [0, 1])
        head_inj = c3.selectbox("إصابة رأس سابقة", [0, 1])
        hyper = c1.selectbox("ضغط الدم المرتفع", [0, 1])
        sys_bp = c2.number_input("ضغط الدم الانقباضي", 90, 180, 120)
        dia_bp = c3.number_input("ضغط الدم الانبساطي", 60, 110, 80)

    with st.expander("🧠 الاختبارات المعرفية والسلوكية"):
        cc1, cc2, cc3 = st.columns(3)
        mmse = cc1.number_input("اختبار MMSE", 0.0, 30.0, 20.0)
        func_ass = cc2.number_input("التقييم الوظيفي", 0.0, 10.0, 5.0)
        mem_comp = cc3.selectbox("شكاوى الذاكرة", [0, 1])
        beh_prob = cc1.selectbox("مشاكل سلوكية", [0, 1])
        adl = cc2.number_input("أنشطة الحياة اليومية (ADL)", 0.0, 10.0, 5.0)
        confusion = cc3.selectbox("الارتباك", [0, 1])
        disorient = cc1.selectbox("فقدان الاتجاه", [0, 1])
        person_ch = cc2.selectbox("تغيرات الشخصية", [0, 1])
        diff_comp = cc3.selectbox("صعوبة إتمام المهام", [0, 1])
        forget = cc1.selectbox("النسيان الشديد", [0, 1])

    # تجميع البيانات لإرسالها للموديل
    # ملاحظة: يجب أن يكون الترتيب مطابقاً تماماً لترتيب الأعمدة في ملف الـ CSV
    features = [
        age, gender, ethnicity, edu, bmi, smoking, alcohol, physical, diet, sleep,
        family_h, cardio, diabetes, depress, head_inj, hyper, sys_bp, dia_bp,
        240, 100, 50, 150, # قيم افتراضية للكوليسترول (CholesterolTotal, LDL, HDL, Trig)
        mmse, func_ass, mem_comp, beh_prob, adl, confusion, disorient, 
        person_ch, diff_comp, forget
    ]

    st.markdown("---")
    if st.button("تحليل البيانات وإصدار التقرير", type="primary"):
        prediction = model.predict([features])
        probability = model.predict_proba([features])[0][1]

        if prediction[0] == 1:
            st.error(f"🚨 النتيجة: احتمالية إصابة بالزهايمر (نسبة التأكد: {probability:.1%})")
            st.warning("ينصح بمراجعة طبيب مختص فوراً لمزيد من الفحوصات.")
        else:
            st.success(f"✅ النتيجة: لا توجد مؤشرات قوية على الإصابة (نسبة الثقة: {1-probability:.1%})")