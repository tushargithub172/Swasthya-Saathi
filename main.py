from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import sqlite3
from flask import redirect, url_for, flash
from fpdf import FPDF
from io import BytesIO
import re
from flask import send_file

app = Flask(__name__)

app.secret_key = '3f8w9j39gwl23@#!t1a2m1kfjw02jl9jjkx6w'

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
model = pickle.load(open('models/random.pkl', 'rb')) # for randomForest

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Symptom and disease mappings
symptoms_dict = { 'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131 }

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    invalid_symptoms = [s for s in patient_symptoms if s not in symptoms_dict]
    if invalid_symptoms:
        raise ValueError(f"⚠ Invalid symptom(s): {', '.join(invalid_symptoms)}")

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1

    prediction = model.predict([input_vector])[0]
    return diseases_list[prediction]

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the "Re-Search" button was clicked
        if 'reset' in request.form:
            return render_template('index.html')  # Reset to the initial form state

        symptoms = request.form.get('symptoms')

        if not symptoms or symptoms.strip() == "" or symptoms == "Symptoms":
            return render_template('index.html')  # Simply reload the page with empty state

        user_symptoms = [s.strip("[]' ").lower().replace(" ", "_") for s in symptoms.split(',') if s.strip()]

        try:
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, pre, medications, rec_diet, workout = helper(predicted_disease)
            my_precautions = [i for i in pre[0]]
            input_summary = f"Our AI System Results predict for symptoms: {', '.join(user_symptoms)}"

            return render_template('index.html',
                                   predicted_disease=predicted_disease,
                                   dis_des=dis_des,
                                   my_precautions=my_precautions,
                                   medications=medications,
                                   my_diet=rec_diet,
                                   workout=workout,
                                   input_summary=input_summary)

        except ValueError as e:
            return render_template('index.html', message=str(e))

    return render_template('index.html')

@app.route('/view_report', methods=['POST'])
def view_report():
    predicted_disease = request.form['predicted_disease']
    user_symptoms = request.form['user_symptoms'].split(', ')
    dis_des = request.form['dis_des']
    my_precautions = request.form['my_precautions'].split(', ')
    medications = request.form['medications'].split(', ')
    my_diet = request.form['my_diet'].split(', ')
    workout = request.form['workout'].split(', ')

    return render_template(
        'report_template.html',
        predicted_disease=predicted_disease,
        user_symptoms=user_symptoms,
        dis_des=dis_des,
        my_precautions=my_precautions,
        medications=medications,
        my_diet=my_diet,
        workout=workout
    )

@app.route('/download', methods=['POST'])
def download_report():
    prediction = request.form['prediction']
    symptoms = request.form['symptoms']
    description = request.form['description']
    precautions = request.form['precautions']
    medications = request.form['medications']
    diet = request.form['diet']
    workout = request.form['workout']

    # Initialize PDF first
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Helper to remove emojis
    def remove_emojis(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    # Section title formatting
    def section_title(title):
        title = remove_emojis(title)
        pdf.set_font("Arial", 'B', 13)
        pdf.set_text_color(21, 115, 71)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=12)

    # Multiline text section
    def write_multiline(label, text):
        section_title(label)
        pdf.multi_cell(0, 10, remove_emojis(text))
        pdf.ln(4)

    # Title bar
    pdf.set_fill_color(21, 115, 71)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "  Swasthya Saathi Medical Report", ln=True, fill=True)

    # Reset font
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    # Fill in report sections
    write_multiline("🩺 Predicted Disease:", prediction)
    write_multiline("🧾 Symptoms Provided:", symptoms)
    write_multiline("📖 Disease Description:", description)
    write_multiline("⚠️ Precautions:", precautions)
    write_multiline("💊 Medications:", medications)
    write_multiline("🥗 Recommended Diet:", diet)
    write_multiline("🏃 Workout Advice:", workout)

    # Emergency section
    section_title("📞 Emergency Contact:")
    pdf.cell(0, 10, "108 (or your local hospital)", ln=True)

    # Output PDF
    pdf_output = BytesIO(pdf.output(dest='S').encode('latin1'))

    return send_file(
        pdf_output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='Swasthya_Saathi_Report.pdf'
    )

import os
import csv
import sqlite3
from datetime import datetime

USE_SQLITE = os.environ.get("USE_SQLITE", "true").lower() == "true"

@app.route('/consultancy', methods=['GET', 'POST'])
def consultancy():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        location = request.form['location']
        concern = request.form['concern']

        if USE_SQLITE:
            # Local development: Use SQLite
            conn = sqlite3.connect('consultancy.db')
            c = conn.cursor()
            c.execute("INSERT INTO consultancy (name, email, phone, location, concern) VALUES (?, ?, ?, ?, ?)",
                      (name, email, phone, location, concern))
            conn.commit()
            conn.close()
        else:
            # Use CSV logging for production environment
            with open('consultancy_logs.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([datetime.now(), name, email, phone, location, concern])

        flash("Your consultancy request has been submitted!", "success")
        return redirect(url_for('consultancy'))

    return render_template('consultancy.html')

# Function to initialize the database, only run once
def init_db():
    conn = sqlite3.connect('consultancy.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS consultancy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            location TEXT NOT NULL,
            concern TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Call init_db() to initialize the database
init_db()  # Run this function only once to create the database table


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True)

