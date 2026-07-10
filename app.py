import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="🌾 CROP RECOMMENDATION",
    page_icon="🌾",
    layout="wide"
)


# ================= BACKGROUND =================

BACKGROUND = "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=2000&q=80"


# ================= CSS =================

st.markdown(f"""

<style>

.stApp {{

background-image:url("{BACKGROUND}");
background-size:cover;
background-position:center;
background-attachment:fixed;

}}


[data-testid="stHeader"] {{

background:rgba(0,0,0,0);

}}


.block-container {{

background:rgba(0,0,0,0.40);
backdrop-filter:blur(18px);
border-radius:25px;
padding:2rem;

}}



.main-title {{

font-size:58px;
font-weight:900;
text-align:center;
color:white;
text-shadow:0px 0px 25px #00ff99;
animation:glow 2s infinite alternate;

}}



@keyframes glow {{

from {{

text-shadow:0px 0px 8px #00ff99;

}}

to {{

text-shadow:0px 0px 30px yellow;

}}

}}



.subtitle {{

text-align:center;
font-size:22px;
color:white;
font-weight:bold;

}}



.card {{

background:rgba(255,255,255,0.15);
backdrop-filter:blur(20px);
padding:25px;
border-radius:25px;
border:1px solid rgba(255,255,255,0.25);
box-shadow:0px 10px 35px rgba(0,0,0,0.4);

}}



div.stButton > button {{

width:100%;
height:65px;
font-size:24px;
font-weight:bold;
border-radius:50px;
border:none;
background:linear-gradient(90deg,#00c853,#64dd17);
color:white;
transition:0.3s;

}}



div.stButton > button:hover {{

transform:scale(1.05);
box-shadow:0px 0px 30px lime;

}}



label,p,h1,h2,h3,h4 {{

color:white!important;
font-weight:bold!important;

}}



.stNumberInput input {{

background:rgba(255,255,255,0.15)!important;
color:white!important;

}}



section[data-testid="stSidebar"] {{

background:rgba(0,0,0,0.55);

}}


</style>

""", unsafe_allow_html=True)



# ================= LOAD MODEL =================


@st.cache_resource
def load_assets():

    try:

        model = joblib.load("crop_recommendation_model.pkl")

        encoder = joblib.load("label_encoder.pkl")

        return model, encoder

    except Exception as e:

        st.error("❌ Model files not found!")

        st.write(e)

        st.stop()



model, encoder = load_assets()
# ================= SIDEBAR =================


with st.sidebar:

    st.markdown("## 🌾 CROP RECOMMENDATION")

    st.success("CROP RECOMMENDATION")


    st.markdown("---")


    st.markdown("### ✅ Features")


    st.write("🌱 Soil Analysis")

    st.write("🌤 Weather Analysis")

    st.write("🤖 AI Prediction")

    st.write("📊 Confidence Score")


    st.markdown("---")


    st.image(
        "https://cdn-icons-png.flaticon.com/512/2909/2909763.png",
        width=140
    )



# ================= HEADER =================


st.markdown(
"""
<div class="main-title">

🌾 SmartCrop AI Dashboard

</div>
""",
unsafe_allow_html=True
)



st.markdown(
"""
<div class="subtitle">

Smart Farming using Artificial Intelligence

</div>
""",
unsafe_allow_html=True
)



st.markdown("---")



# ================= INPUT SECTION =================


left,right = st.columns(
    [1.1,1],
    gap="large"
)



with left:


    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )


    st.subheader("🌱 Soil Parameters")



    c1,c2,c3 = st.columns(3)



    with c1:

        nitrogen = st.number_input(
            "Nitrogen",
            min_value=0,
            max_value=200,
            value=50,
            step=1
        )



    with c2:

        phosphorus = st.number_input(
            "Phosphorus",
            min_value=0,
            max_value=200,
            value=50,
            step=1
        )



    with c3:

        potassium = st.number_input(
            "Potassium",
            min_value=0,
            max_value=200,
            value=50,
            step=1
        )



    st.write("")



    st.subheader("🌤 Weather Parameters")



    temperature = st.slider(
        "Temperature (°C)",
        0,
        50,
        26
    )



    humidity = st.slider(
        "Humidity (%)",
        0,
        100,
        70
    )



    ph = st.slider(
        "Soil pH",
        0,
        14,
        6
    )



    rainfall = st.slider(
        "Rainfall (mm)",
        0,
        500,
        120
    )



    predict = st.button(
        "🚀 Predict Best Crop"
    )



    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )



# ================= VISUALIZATION AREA =================


with right:


    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )


    st.subheader("📊 AI Soil Visualization")



    fig = go.Figure()



    fig.add_trace(

        go.Scatterpolar(

            r=[

                nitrogen,

                phosphorus,

                potassium,

                ph*10

            ],


            theta=[

                "Nitrogen",

                "Phosphorus",

                "Potassium",

                "pH ×10"

            ],


            fill="toself",


            line=dict(

                color="#00ff99",

                width=4

            )

        )

    )



    fig.update_layout(

        height=450,

        paper_bgcolor="rgba(0,0,0,0)",

        polar=dict(

            bgcolor="rgba(0,0,0,0)",


            radialaxis=dict(

                visible=True,

                range=[0,200],

                tickfont=dict(color="white")

            ),


            angularaxis=dict(

                tickfont=dict(color="white")

            )

        ),


        showlegend=False

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )



    st.markdown(

        "</div>",

        unsafe_allow_html=True

    )
    # ================= PREDICTION =================


if predict:


    progress = st.progress(0)

    status = st.empty()



    for i in range(101):

        time.sleep(0.01)

        progress.progress(i)

        status.markdown(
            f"### 🤖 AI Processing... {i}%"
        )



    status.empty()



    # ================= INPUT DATA =================


    sample = pd.DataFrame(

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

            "Nitrogen",

            "Phosphorus",

            "Potassium",

            "Temperature",

            "Humidity",

            "pH_Value",

            "Rainfall"

        ]

    )



    try:


        prediction = model.predict(sample)


        crop = encoder.inverse_transform(prediction)[0]



        if hasattr(model,"predict_proba"):

            confidence = (

                model.predict_proba(sample).max()

                *100

            )

        else:

            confidence = 95



    except Exception as e:


        st.error(
            "Prediction Error"
        )

        st.write(e)

        st.stop()



    # ================= RESULT CARD =================


    st.markdown(
    f"""

    <div style="

    background:linear-gradient(135deg,#11998e,#38ef7d);

    border-radius:25px;

    padding:35px;

    text-align:center;

    color:white;

    box-shadow:0px 15px 35px rgba(0,0,0,0.4);

    animation:fade 1s;

    ">


    <h3>

    🌾 Recommended Crop

    </h3>



    <h1 style="

    font-size:60px;

    text-shadow:0px 0px 20px white;

    ">

    {crop.upper()}

    </h1>



    </div>



    <style>


    @keyframes fade{{


    from{{

    opacity:0;

    transform:translateY(40px);

    }}


    to{{

    opacity:1;

    transform:translateY(0);

    }}


    }}


    </style>


    """,

    unsafe_allow_html=True

    )



    # ================= CROP IMAGES =================


    crop_images = {


        "rice":
        "https://images.unsplash.com/photo-1536657464919-892534f60d6e?w=900",


        "maize":
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=900",


        "cotton":
        "https://images.unsplash.com/photo-1592928302636-c83cf1e1f8e4?w=900",


        "coffee":
        "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=900",


        "banana":
        "https://images.unsplash.com/photo-1603833665858-e61d17a86224?w=900",


        "apple":
        "https://images.unsplash.com/photo-1567306226416-28f0efdc88ce?w=900",


        "mango":
        "https://images.unsplash.com/photo-1553279768-865429fa0078?w=900",


        "grapes":
        "https://images.unsplash.com/photo-1515778767554-4244b5f9d5b6?w=900"

    }



    crop_name = crop.lower()



    if crop_name in crop_images:


        st.image(

            crop_images[crop_name],

            caption=f"🌱 {crop} Plantation",

            use_container_width=True

        )



    # ================= CONFIDENCE GAUGE =================


    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",


            value=confidence,


            number={

                "suffix":"%"

            },


            title={

                "text":

                "Prediction Confidence"

            },


            gauge={


                "axis":{

                    "range":[0,100]

                },


                "bar":{

                    "color":"lime"

                },


                "steps":[


                    {

                    "range":[0,40],

                    "color":"red"

                    },


                    {

                    "range":[40,70],

                    "color":"yellow"

                    },


                    {

                    "range":[70,100],

                    "color":"green"

                    }


                ]

            }

        )

    )



    gauge.update_layout(

        height=350,

        paper_bgcolor="rgba(0,0,0,0)",

        font=dict(color="white")

    )



    st.plotly_chart(

        gauge,

        use_container_width=True

    )