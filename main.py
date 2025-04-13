import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import csv
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import gaussian_kde
from datetime import date
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

query_params = st.query_params
if "user" in query_params and query_params["user"]:
    st.session_state.logged_in = True
    st.session_state.current_user = query_params["user"]
else:
    st.session_state.logged_in = False
    st.session_state.current_user = None

if not st.session_state.logged_in:
    with st.sidebar:
        selected = option_menu("Main Menu", ["Login", "Register"], icons=["box-arrow-in-right", "person-plus"],menu_icon=["cast"], default_index=0, orientation="vertical",
                               styles={"nav-link-selected": {
                                                "background-color": "#00A36C",
                                                "color": "white",
                                                "border-radius": "8px"
                                }})
    st.markdown("""
        <style>
            .stButton>button {background-color: #00A36C; color: white; width: 100%; border-radius: 10px;}
            div[data-baseweb="input"]:focus-within{border-color: #00A36C !important;}
            .stButton>button:hover {border: 2px solid #F5F5DC !important;color: #F5F5DC !important;}
        </style>
        """, unsafe_allow_html=True)
    if not st.session_state.current_user:
        if selected == "Login":
            st.markdown(
                """
                <style>
                .centered-title {
                    text-align: center;
                }
                </style>
                <h1 class="centered-title">Login Form</h1>
                """,
                unsafe_allow_html=True
            )
            login_name = st.text_input("Name")
            login_password = st.text_input("Password", type="password")

            submit = st.button(label="Login")

            df = pd.read_csv("reg_data.csv")

            if submit:
                if not login_name or not login_password:
                    st.error("Please enter both email and password.")
                elif ((df["Name"] == login_name) & (df['Password'] == login_password)).any():
                    st.session_state.logged_in=True
                    st.session_state.current_user=login_name
                    st.query_params["user"] = login_name
                    st.rerun()
                else:
                    st.error("Invalid username or password.Please try again")
        elif selected == "Register":
            st.markdown(
                    """
                    <style>
                    .centered-title {
                        text-align: center;
                    }
                    </style>
                    <h1 class="centered-title">Registration Form</h1>
                    """,
                    unsafe_allow_html=True
                )

            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")

            submit = st.button(label="Register")

            df = pd.read_csv("reg_data.csv")

            if submit:
                if name in df["Name"].values:
                    st.error("Username already taken. Choose another.")
                elif not name or not email or not password:
                    st.error("All fields are required!")
                elif not email.endswith("@gmail.com") or "@" not in email:
                    st.error("Please enter a valid Gmail address (example@gmail.com)")
                elif len(password) < 8 or not any(c.islower() for c in password) or not any(
                        c.isupper() for c in password) or not any(c.isdigit() for c in password) or not any(
                        c in "@$!%*?&" for c in password):
                    st.error("Password must be at least 8 characters long and include at least one uppercase letter, one lowercase letter, one number, and one special symbol.")
                else:
                    fields = [name, email, password]
                    with open("reg_data.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)
                    st.success(f"Registration successful! Welcome, {name}!")

if st.session_state.logged_in:
    with st.sidebar:
        selected = option_menu("Home", ["Dashboard", "Risk Assessment","Diabetes Prediction","Heart Attack Prediction","Disease Outbreak Forecasting","Smoking Analytics","Help & Support","About us","Log out"],icons=["cast", "hospital","graph-up","activity","graph-up","bar-chart","question-circle","people", "box-arrow-right"], menu_icon=["house"],default_index=0, orientation="vertical",
                               styles={"nav-link-selected": {
                                   "background-color": "#00A36C",
                                   "color": "white",
                                   "border-radius": "8px"
                               }})
    if selected=="Dashboard":
        st.header(f"Welcome ,{st.session_state.current_user}!")

        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            div[data-baseweb="select"]:focus-within > div {
            border-color: #00A36C !important;
            }
            </style>
            <h1 class="centered-title">Health Risk Assessment & Disease Forecasting Portal</h1>
            """,
            unsafe_allow_html=True
        )

        option=st.selectbox("Select Analysis Type",["Blood type analytics","Number of people affected by each disease","Age wise health risk","Symptom analytics","Correlation Heatmap"])

        df = pd.read_csv("patient_dataset.csv")

        if option=="Blood type analytics":
            st.markdown(
                """
                <style>
                .centered-title {
                    text-align: center;
                }
                </style>
                <h3 class="centered-title">Blood type analytics</h3>
                """,
                unsafe_allow_html=True
            )
            male = df[df["Gender"] == "Male"]["Blood Type"].value_counts()
            female = df[df["Gender"] == "Female"]["Blood Type"].value_counts()

            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                subplot_titles=["Male", "Female"]
            )

            fig.add_trace(go.Pie(
                labels=male.index,
                values=male.values,
                name="Male",
                hole=0.3,
                hoverinfo="label+percent+value"
            ), row=1, col=1)

            fig.add_trace(go.Pie(
                labels=female.index,
                values=female.values,
                name="Female",
                hole=0.3,
                hoverinfo="label+percent+value"
            ), row=1, col=2)

            fig.update_layout(
                margin=dict(l=40, r=40, t=30, b=40),
                height=400,
                template="plotly_white",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
        elif option=="Number of people affected by each disease":
            st.markdown(
                """
                <style>
                .centered-title {
                    text-align: center;
                }
                </style>
                <h3 class="centered-title">Number of people affected by each disease</h3>
                """,
                unsafe_allow_html=True
            )
            disease_list = df["Disease"].str.split(", ").explode()
            disease_counts = disease_list.value_counts()
            fig = go.Figure(go.Bar(
                x=disease_counts.values,
                y=disease_counts.index,
                orientation='h',
                text=disease_counts.values,
                textposition='auto',
            ))

            fig.update_layout(
                xaxis_title="Number of people",
                yaxis_title="Disease",
                template="plotly_white",
                height=600,
                margin=dict(l=40, r=40, t=30, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
        elif option=="Age wise health risk":
            st.markdown(
                """
                <style>
                .centered-title {
                    text-align: center;
                }
                </style>
                <h3 class="centered-title">Age Wise Health Risk Analytics</h3>
                """,
                unsafe_allow_html=True
            )
            df["Health Risk"] = (df["Age"] / 10) + (df["Disease"] != "No Disease") * 5 + (
                        df["Disability"] != "No Disability") * 3
            age_data = df["Age"].dropna()
            hist_data = [df["Age"]]
            group_labels = ['Age']

            fig = ff.create_distplot(
                hist_data,
                group_labels,
                bin_size=2,
                show_rug=False,
                curve_type='kde',
            )

            kde = gaussian_kde(age_data, bw_method=0.2)
            x_vals = np.linspace(0, age_data.max(), 500)
            y_vals = kde(x_vals)

            fig.data = tuple(trace for i, trace in enumerate(fig.data) if i != 1)

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name='KDE (smooth)',
                line=dict(color='darkblue', width=3)
            ))

            fig.update_layout(
                xaxis_title="Age",
                yaxis_title="Health Risk Score",
                template='plotly_white',
                height=500,
                margin=dict(l=40, r=40, t=30, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
        elif option=="Symptom analytics":
            st.markdown(
                """
                <style>
                .centered-title {
                    text-align: center;
                }
                </style>
                <h3 class="centered-title">Symptom Frequency Distribution</h3>
                """,
                unsafe_allow_html=True
            )
            symptom_list = df["Symptom"].dropna().str.split(", ").explode()
            symptom_freq = symptom_list.value_counts()
            fig = go.Figure(data=[
                go.Bar(
                    x=symptom_freq.index,
                    y=symptom_freq.values,
                )
            ])

            fig.update_layout(
                xaxis_title="Symptoms",
                yaxis_title="Frequency",
                template="plotly_white",
                height=500,
                margin=dict(l=40, r=40, t=30, b=40)
            )

            fig.update_xaxes(tickangle=300)

            st.plotly_chart(fig, use_container_width=True)
        elif option=="Correlation Heatmap":
            st.markdown(
                """
                <style>
                .centered-title {
                    text-align: center;
                }
                </style>
                <h3 class="centered-title">Correlation Heatmap</h3>
                """,
                unsafe_allow_html=True
            )
            df["Symptom Count"] = df["Symptom"].dropna().str.split(", ").apply(len)
            subset_corr = df[["Age", "Symptom Count"]].corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=subset_corr.values,
                    x=subset_corr.columns,
                    y=subset_corr.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=subset_corr.round(2).values,
                    hovertemplate='%{y} vs %{x}: %{z:.2f}<extra></extra>',
                    showscale=True,
                    colorbar=dict(title='Correlation')
                )
            )

            fig.update_layout(
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=40, t=30, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
    elif selected == "Risk Assessment":
        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h2 class="centered-title">Patient Details Form</h2>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
            <style>
                .stButton>button {background-color: #00A36C; color: white; width: 100%; border-radius: 10px;}
                div[data-baseweb="input"]:focus-within{border-color: #00A36C !important;}
                .stButton>button:hover {border: 2px solid #F5F5DC !important;color: #F5F5DC !important;}
                div[data-baseweb="select"]:focus-within > div {border-color: #00A36C !important;}
                div[aria-label*="Selected"]::after{background-color: #00A36C !important;color: white !important;border-radius: 50%;}
                [data-testid="stRadio"] > div > label > div:first-child{background-color: #00A36C !important;}
            </style>
        """, unsafe_allow_html=True)

        df = pd.read_csv("patient_dataset.csv")

        df.drop(columns=["Name", "Date"], inplace=True)

        gender_encode = {"Male": 1, "Female": 0, "Other": 2}
        df["Gender"] = df["Gender"].map(gender_encode)

        blood_type_encode = {"O+": 1, "O-": 2, "A+": 3, "A-": 4, "B+": 5, "B-": 6, "AB+": 7, "AB-": 8}
        df["Blood Type"] = df["Blood Type"].map(blood_type_encode)

        smoking_encode = {"Non-Smoker": 0, "Occasional": 1, "Regular Smoker": 2}
        df["Smoking"] = df["Smoking"].map(smoking_encode)

        alcohol_encode = {"Never": 0, "Occasionally": 1, "Frequently": 2}
        df["Alcohol"] = df["Alcohol"].map(alcohol_encode)

        dietary_encode = {"Vegetarian": 0, "Non-Vegetarian": 1, "Vegan": 2, "Other": 3}
        df["Diet"] = df["Diet"].map(dietary_encode)

        activity_encode = {"Sedentary": 0, "Moderate": 1, "Active": 2, "Highly Active": 3}
        df["Activity Level"] = df["Activity Level"].map(activity_encode)

        disability_encode = {"No disability": 0, "Visual Impairment": 1, "Hearing Impairment": 2,
                             "Mobility Impairment": 3,
                             "Cognitive Disability": 4, "Mental Health Condition": 5, "Other": 6}
        df["Disability"] = df["Disability"].map(disability_encode)

        df = pd.get_dummies(df, columns=["Symptom"])

        X = df.drop(columns=["Disease"])
        y = df["Disease"]

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)

        def predict_disease(user_input):
            user_input["Gender"] = gender_encode.get(user_input["Gender"], -1)
            user_input["Blood Type"] = blood_type_encode.get(user_input["Blood Type"], -1)
            user_input["Smoking"] = smoking_encode.get(user_input["Smoking"], -1)
            user_input["Alcohol"] = alcohol_encode.get(user_input["Alcohol"], -1)
            user_input["Diet"] = dietary_encode.get(user_input["Diet"], -1)
            user_input["Activity Level"] = activity_encode.get(user_input["Activity Level"], -1)
            user_input["Disability"] = disability_encode.get(user_input["Disability"], -1)

            symptom_columns = [col for col in X.columns if "Symptom_" in col]
            symptom_data = {col: 0 for col in symptom_columns}
            for symptom in user_input["Symptom"]:
                symptom_col = f"Symptom_{symptom}"
                if symptom_col in symptom_data:
                    symptom_data[symptom_col] = 1

            input_df = pd.DataFrame([{**user_input, **symptom_data}])
            input_df = input_df[X.columns]

            prediction = model.predict(input_df)
            return prediction[0]

        name = st.text_input("Full Name")
        today = date.today()
        login_date = st.date_input("Date", min_value=today, max_value=today, value=today)
        age = st.text_input("Age")
        gender= st.radio("Gender",["Male","Female","Other"])
        disease=st.multiselect("Disease",["Cold","Covid","Fever","Thalassemia","Cancer","Diabeties","Brain tumor","Malaria","Dengue","Migraine","AIDS","Heart attack","No Disease"])
        symptom=st.multiselect("Symptoms",["Cold","Fatigue", "Weight Loss", "Fever", "Night Sweats", "Headache", "Nausea", "Seizures", "Vision Problems","Pain", "Lumps", "Cough", "Runny Nose", "Sore Throat", "Sneezing", "Shortness of Breath", "Loss of Taste/Smell","Muscle Pain", "Rash", "Increased Thirst", "Frequent Urination", "Blurred Vision", "Chills", "Sweating","Body Aches", "Chest Pain", "Dizziness", "Severe Headache", "Light Sensitivity", "Weakness", "Pale Skin"])
        blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
        smoking = st.radio("Smoking Status", ["Non-Smoker", "Occasional", "Regular Smoker"])
        alcohol = st.radio("Alcohol Consumption", ["Never", "Occasionally", "Frequently"])
        diet = st.selectbox("Dietary Preferences", ["Vegetarian", "Vegan", "Non-Vegetarian", "Other"])
        activity_level = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "Active", "Highly Active"])
        disability = st.selectbox("Select Disability Type", ["No disability","Visual Impairment","Hearing Impairment","Mobility Impairment","Cognitive Disability","Mental Health Condition","Other"])

        submit = st.button("Submit")

        if submit:
            if not name or not age:
                st.error("All fields required!")
            else:
                formatted_date = login_date.strftime("%Y/%m/%d")
                disease_str = ", ".join(disease) if disease else "No Disease"
                symptom_str = ", ".join(symptom) if symptom else "None"
                fields = [name,formatted_date,age, gender,disease_str,symptom_str, blood_type, smoking, alcohol, diet, activity_level,disability]
                with open("patient_dataset.csv",  mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                user_data = {
                    "Age": int(age),
                    "Gender": gender,
                    "Blood Type": blood_type,
                    "Smoking": smoking,
                    "Alcohol": alcohol,
                    "Diet": diet,
                    "Activity Level": activity_level,
                    "Disability": disability,
                    "Symptom": symptom_str,
                    "Disease":disease_str
                }
                predicted_disease = predict_disease(user_data)
                st.success(f"Predicted Disease: {predicted_disease}")
                if disease:
                    st.info(f"User-reported Disease(s): {', '.join(disease)}")
    elif selected=="Diabetes Prediction":
        st.markdown(
            """
            <style>
            .centered-title {text-align: center;}
            </style>
            <h1 class="centered-title">Diabetes Risk Prediction</h1>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
            <style>
                .stButton>button {background-color: #00A36C; color: white; width: 100%; border-radius: 10px;}
                .stButton>button:hover {border: 2px solid #F5F5DC !important;color: #F5F5DC !important;}
                 div[data-testid="stNumberInput"] > div:has(input:focus){border: 2px solid #00A86B !important;}
            </style>
        """, unsafe_allow_html=True)
        st.write("Enter the following details to predict diabetes risk:")
        st.number_input("Pregnancy", step=1, min_value=0)
        blood_glucose_level = st.number_input("Glucose Level", min_value=0)
        st.number_input("Blood Pressure", min_value=0.00)
        st.number_input("Skin Thickness", min_value=0.00)
        st.number_input("Insulin Level", min_value=0.00)
        bmi = st.number_input("BMI", min_value=0.00)
        st.number_input("Diabetes Pedigree Function", min_value=0.00)
        age = st.number_input("Age", min_value=0)
        submit = st.button("Predict")

        df = pd.read_csv("diabetes_prediction_dataset.csv")

        features = ["blood_glucose_level", "bmi", "age"]
        input = df[features]
        target = df["diabetes"]

        dt = DecisionTreeClassifier()
        dt.fit(input, target)

        if submit:
            input_data = [blood_glucose_level, bmi, age]
            prediction = dt.predict([input_data])[0]
            st.success(f"Prediction: {'Diabetic ' if prediction == 1 else 'Not Diabetic'}")

        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h3 class="centered-title">Diabetic analytics</h3>
            """,
            unsafe_allow_html=True
        )

        diabetes_counts = df["diabetes"].value_counts()
        labels = ["Non-Diabetic", "Diabetic"]

        fig = go.Figure(
            data=[go.Pie(
                labels=labels,
                values=diabetes_counts.values,
                textinfo='percent+label',
                hole=0.3,
                rotation=140
            )]
        )

        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    elif selected=="Heart Attack Prediction":
        st.markdown(
            """
            <style>
            .centered-title {text-align: center;}
            </style>
            <h1 class="centered-title">Heart Attack Prediction</h1>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
                    <style>
                        .stButton>button {background-color: #00A36C; color: white; width: 100%; border-radius: 10px;}
                        .stButton>button:hover {border: 2px solid #F5F5DC !important;color: #F5F5DC !important;}
                         div[data-testid="stNumberInput"] > div:has(input:focus){border: 2px solid #00A86B !important;}
                         div[data-baseweb="select"]:focus-within > div {border-color: #00A36C !important;}
                         [data-testid="stRadio"] > div > label > div:first-child{background-color: #00A36C !important;}
                    </style>
                """, unsafe_allow_html=True)
        df = pd.read_csv("stroke_data.csv")

        df = df[["gender", "age", "hypertension", "heart_disease", "smoking_status", "stroke"]]
        df = df.dropna()

        encoder_gender = LabelEncoder()
        encoder_smoking = LabelEncoder()

        df["gender"] = encoder_gender.fit_transform(df["gender"])
        df["smoking_status"] = encoder_smoking.fit_transform(df["smoking_status"])

        X = df.drop(columns=["stroke"])
        y = df["stroke"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        gender_options = list(encoder_gender.classes_)
        smoking_status_options = list(encoder_smoking.classes_)

        gender = st.selectbox("Gender", gender_options)
        age = st.number_input("Age", min_value=0, max_value=120)
        hypertension = st.radio("Hypertension", ["Yes", "No"])
        heart_disease = st.radio("Heart Disease", ["Yes", "No"])
        smoking_status = st.selectbox("Smoking Status", smoking_status_options)

        gender_encoded = encoder_gender.transform([gender])[0]
        smoking_status_encoded = encoder_smoking.transform([smoking_status])[0]
        hypertension_encoded = 1 if hypertension == "Yes" else 0
        heart_disease_encoded = 1 if heart_disease == "Yes" else 0

        user_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, smoking_status_encoded]])
        prediction = model.predict(user_data)[0]
        prediction_proba = model.predict_proba(user_data)[0][1] * 100

        submit = st.button("Predict")

        if submit:
            if prediction == 1:
                st.error(f"High Risk of Heart Attack! ({prediction_proba:.2f}% probability)")
            else:
                st.success(f"Low Risk of Heart Attack ({100 - prediction_proba:.2f}% probability)")
    elif selected=="Disease Outbreak Forecasting":
        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h1 class="centered-title">Disease Outbreak Trend Visualization</h1>
            """,
            unsafe_allow_html=True
        )
        df = pd.read_csv("patient_dataset.csv")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
        df_filtered = df[df["Disease"] != "No Disease"]

        df_daily_cases = df_filtered.groupby("Date").size().reset_index(name="Number of Cases")
        df_daily_cases["Date"] = pd.to_datetime(df_daily_cases["Date"])
        df_monthly_cases = df_daily_cases.set_index("Date").resample("M").sum().reset_index()

        fig = px.line(
            df_monthly_cases,
            x="Date",
            y="Number of Cases",
            markers=True,
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Cases",
            xaxis_tickformat="%b %Y",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
            height=450
        )

        fig.update_xaxes(tickangle=300,dtick="M1")

        st.plotly_chart(fig, use_container_width=True)
    elif selected=="Smoking Analytics":
        file_path = "stroke_data.csv"
        df = pd.read_csv(file_path)

        male = df[df["gender"] == "Male"]["smoking_status"].value_counts()
        female = df[df["gender"] == "Female"]["smoking_status"].value_counts()

        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h1 class="centered-title">Smoking Status Analytics by Gender</h1>
            """,
            unsafe_allow_html=True
        )

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=["Male", "Female"]
        )

        fig.add_trace(go.Pie(
            labels=male.index,
            values=male.values,
            name="Male",
            hole=0.3,
            hoverinfo="label+percent+value",
        ), row=1, col=1)

        fig.add_trace(go.Pie(
            labels=female.index,
            values=female.values,
            name="Female",
            hole=0.3,
            hoverinfo="label+percent+value",
        ), row=1, col=2)

        fig.update_layout(
            margin=dict(l=40, r=40, t=30, b=40),
            height=400,
            template="plotly_white",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        df = df[["gender", "age", "smoking_status"]].dropna()

        df["age_group"] = (df["age"] // 10) * 10

        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h1 class="centered-title">Age-wise Smoking Status Analytics</h1>
            """,
            unsafe_allow_html=True
        )

        male_df = df[df["gender"] == "Male"].groupby("age_group")["smoking_status"].value_counts().reset_index(
            name="count")
        female_df = df[df["gender"] == "Female"].groupby("age_group")["smoking_status"].value_counts().reset_index(
            name="count")

        fig_male = px.bar(male_df,
                          x="age_group",
                          y="count",
                          color="smoking_status",
                          title="Male Smoking Status Over Age Groups",
                          barmode="group")
        st.plotly_chart(fig_male, use_container_width=True)

        fig_female = px.bar(female_df,
                            x="age_group",
                            y="count",
                            color="smoking_status",
                            title="Female Smoking Status Over Age Groups",
                            barmode="group")
        st.plotly_chart(fig_female, use_container_width=True)
    elif selected=="Help & Support":
        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h2 class="centered-title">Help & Support</h2>
            """,
            unsafe_allow_html=True
        )

        st.markdown("""
        Welcome to the **Parameter-Based Health Risk Assessment and Disease Outbreak Forecasting** platform!  
        This tool uses machine learning to assess health risks and predict potential disease outbreaks.  
        If you need help, check out the sections below.  
        """)

        st.header("How to Use the Platform")

        st.markdown("""
        1 **Enter Patient Details:** Provide your Name, Age, Gender, Blood Type, and Lifestyle Factors.  
        2 **Select Symptoms & Diseases:** Choose from the available symptom and disease options.  
        3 **Get Predictions:** Click the "Predict Disease" button for health risk assessment.  
        4 **Interpret Results:** Review the prediction and seek medical advice if necessary.  
        """)

        st.header("Frequently Asked Questions")

        with st.expander("How accurate is the disease prediction?"):
            st.write(
                "The system provides a probable prediction based on historical health data, but it is not a substitute for a doctor's diagnosis.")

        with st.expander("Can I enter multiple symptoms?"):
            st.write("Yes! You can select multiple symptoms that match your condition for a more accurate analysis.")

        with st.expander("Is my data secure?"):
            st.write("Yes, we prioritize data security. Your personal health information is processed securely.")

        with st.expander("Can I use this tool for self-diagnosis?"):
            st.write(
                "No, this tool is meant for **health risk assessment** and **early warning**. Always consult a doctor for confirmation.")
    elif selected == "About us":
        st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
            }
            </style>
            <h2 class="centered-title">About Us</h2>
            """,
            unsafe_allow_html=True
        )
        st.header("Description")
        description = """
            <div style="text-align: justify;">
                The "Parameter-Based Health Risk Assessment and Disease Outbreak
                Forecasting" project is a comprehensive data analytics and machine learning initiative aimed at leveraging
                advanced algorithms and artificial intelligence techniques to assess individual health risks based on
                various parameters and forecast potential disease outbreaks. By integrating diverse data sources, the
                project aims to provide personalized health risk assessments for individuals and contribute to early
                detection and prediction of disease outbreaks on a larger scale.
            </div>
        """
        st.markdown(description, unsafe_allow_html=True)

        st.header("Problems in the Existing System")
        st.write("1. Delayed Disease Detection – Lack of early warning mechanisms.")
        st.write("2. Generalized Health Assessments – One-size-fits-all approaches fail to address individual risks.")
        st.write("3. Inefficient Data Utilization – Health data remains underutilized for prediction.")
        st.write("4. Lack of Real-Time Insights – Absence of predictive analytics in healthcare decision-making.")
        st.write("5. Limited Integration of Diverse Data Sources – Incomplete risk assessment due to isolated data.")

        st.header("Purpose of the Project")
        st.write("To provide personalized health risk assessments based on multiple parameters.")
        st.write("To develop a predictive model for forecasting disease outbreaks.")
        st.write("To enhance preventive healthcare by enabling early intervention.")
        st.write("To integrate real-time health monitoring data for risk evaluation.")
        st.write("To assist healthcare professionals and authorities with data-driven insights.")
    elif selected == "Log out":
        st.query_params.clear()  
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()