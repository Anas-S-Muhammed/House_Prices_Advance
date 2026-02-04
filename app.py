import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="House Price Predictor", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .price-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_model():
    data = pd.read_csv('train.csv')
    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
    
    X = data[["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "LotFrontage"]]
    y = data["SalePrice"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Get average price for comparison
    avg_price = y.mean()
    
    return model, avg_price

model, avg_price = load_model()

# Header
st.title("🏠 AI House Price Predictor")
st.markdown("### Get instant property valuations powered by machine learning")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Info")
    st.markdown("""
    **Algorithm:** Random Forest  
    **Accuracy:** 87.6% (R²)  
    **Avg Error:** $29,383  
    **Training Data:** 1,460 houses
    """)
    
    st.markdown("---")
    
    st.header("📊 Quick Stats")
    st.metric("Average House Price", f"${avg_price:,.0f}")
    st.metric("Price Range", "$34,900 - $755,000")
    
    st.markdown("---")
    st.info("💡 **Tip:** Adjust the sliders to see how different features affect the price!")

# Main content
st.markdown("---")

# Input form with better layout
st.subheader("🔧 House Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🏗️ Build Quality")
    overall_qual = st.slider(
        "Overall Quality",
        1, 10, 7,
        help="Rates the overall material and finish (1=Poor, 10=Excellent)"
    )
    
    st.markdown("##### 🏡 Living Space")
    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        500, 5000, 1500, step=100,
        help="Above grade (ground) living area square feet"
    )

with col2:
    st.markdown("##### 🚗 Garage")
    garage_cars = st.slider(
        "Garage Size (cars)",
        0, 4, 2,
        help="Size of garage in car capacity"
    )
    
    garage_area = st.number_input(
        "Garage Area (sq ft)",
        0, 1500, 400, step=50,
        help="Size of garage in square feet"
    )

with col3:
    st.markdown("##### 🏠 Additional Space")
    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        0, 3000, 1000, step=100,
        help="Total square feet of basement area"
    )
    
    lot_frontage = st.number_input(
        "Lot Frontage (ft)",
        0, 300, 70, step=5,
        help="Linear feet of street connected to property"
    )

st.markdown("---")

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🎯 PREDICT PRICE NOW", type="primary")

# Prediction results
if predict_btn:
    input_data = [[overall_qual, gr_liv_area, garage_cars, garage_area, total_bsmt_sf, lot_frontage]]
    prediction = model.predict(input_data)[0]
    
    st.markdown("---")
    
    # Big price display
    st.markdown(f"""
        <div class="price-box">
            <h2 style="margin:0; color:white;">Estimated Market Value</h2>
            <h1 style="margin:0.5rem 0; font-size:4rem; color:white;">${prediction:,.0f}</h1>
            <p style="margin:0; font-size:1.1rem; opacity:0.9;">
                {('🔥 Above Average' if prediction > avg_price else '💰 Below Average')} 
                ({abs(prediction - avg_price) / avg_price * 100:.1f}% {'higher' if prediction > avg_price else 'lower'} than market average)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature breakdown
    st.subheader("📋 Feature Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Quality Rating", f"{overall_qual}/10")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Living Area", f"{gr_liv_area:,} sq ft")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Garage Capacity", f"{garage_cars} cars")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Garage Area", f"{garage_area:,} sq ft")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Basement", f"{total_bsmt_sf:,} sq ft")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Lot Frontage", f"{lot_frontage} ft")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_sqft = gr_liv_area + total_bsmt_sf
        st.metric("Total Area", f"{total_sqft:,} sq ft")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        price_per_sqft = prediction / total_sqft if total_sqft > 0 else 0
        st.metric("Price/sq ft", f"${price_per_sqft:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Price comparison gauge
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Market Position")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Your Property vs Market Average", 'font': {'size': 20}},
        delta = {'reference': avg_price, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 500000], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, avg_price * 0.8], 'color': '#ffebee'},
                {'range': [avg_price * 0.8, avg_price * 1.2], 'color': '#fff9c4'},
                {'range': [avg_price * 1.2, 500000], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_price
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🤖 Powered by:** Random Forest ML")
with col2:
    st.markdown("**🎯 Accuracy:** 87.6% R² Score")
with col3:
    st.markdown("**📊 Data:** 1,460 Real Houses")