import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt

# Custom Header Styling
st.markdown(
    """
    <style>
    body {
        background-color: #121212; 
        color: #FFFFFF; 
    }
    .custom-header {
        font-size: 2em;
        color: #BB86FC;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] {
        background-color: #1F1B24;
        color: #FFFFFF;
        border-right: 1px solid #BB86FC;
    }
    .stButton > button {
        background-color: #BB86FC;
        color: #121212;
        font-size: 16px;
    }
    .stTextInput > div > input {
        background-color: #1F1B24;
        color: #FFFFFF;
    }
    .stSlider > div > div > div > div {
        background-color: #BB86FC;
    }
    table {
        background-color: #1F1B24;
        color: #FFFFFF;
        border: 1px solid #BB86FC;
    }
    th {
        color: #BB86FC;
    }
    td {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Sales", "Trends", "Product", "Gender & Country Sales", "Prediction"],
        icons=["house", "graph-up-arrow", "bar-chart", "cart", "globe", "calculator"],
        menu_icon="cast",
        default_index=0,
    )

# Function to load and preprocess data
@st.cache_data
def get_data():
    url = "https://raw.githubusercontent.com/Samarth4507/Sales-Predicting-dashboard/main/sales_data.csv"
    try:
        data = pd.read_csv(url)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
        data.dropna(subset=['timestamp'], inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data from the URL: {e}")
        return pd.DataFrame()

# Function to load the pickle model
@st.cache_resource
def load_model():
    try:
        with open('regression_model.pkl', 'rb') as file:
            loaded_content = pickle.load(file)
            # Check if the loaded object is a tuple, and extract the model if so
            if isinstance(loaded_content, tuple):
                model = loaded_content[0]  # Assume the model is the first element
            else:
                model = loaded_content
            return model
    except FileNotFoundError:
        st.error("Model file not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to apply filters
def apply_filters(data):
    st.subheader("Filters")
    product_filter = st.multiselect("Select Product(s)", data['product'].unique(), default=data['product'].unique())
    data = data[data['product'].isin(product_filter)]

    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=data['timestamp'].min().date(),
        max_value=data['timestamp'].max().date(),
        value=(data['timestamp'].min().date(), data['timestamp'].max().date()),
        format="YYYY-MM-DD",
    )
    data = data[(data['timestamp'] >= pd.to_datetime(start_date)) & (data['timestamp'] <= pd.to_datetime(end_date))]

    country_filter = st.multiselect("Select Country(s)", data['Country'].unique(), default=data['Country'].unique())
    data = data[data['Country'].isin(country_filter)]

    gender_filter = st.multiselect("Select Gender(s)", data['gender'].unique(), default=data['gender'].unique())
    data = data[data['gender'].isin(gender_filter)]

    return data

# Load and preprocess the dataset
df = get_data()
required_columns = ['timestamp', 'customer_id', 'gender', 'Country', 'product', 'units_sold', 'price_per_unit', 'total_sales']

# Check if the dataset is loaded correctly and contains the required columns
if df.empty or not all(col in df.columns for col in required_columns):
    st.error("Missing or invalid data. Please check the dataset.")
else:
    filtered_data = apply_filters(df)

    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        if selected == "Home":
            st.title("Home Page")
            st.write("Welcome to the Sales Dashboard!")
            total_sales = filtered_data['total_sales'].sum()
            total_units_sold = filtered_data['units_sold'].sum()

            st.metric(label="Total Sales", value=f"${total_sales:,.2f}")
            st.metric(label="Units Sold", value=f"{total_units_sold:,}")
        elif selected == "Sales":
            st.title("Sales Page")
            sales_over_time = filtered_data.groupby(['timestamp', 'product'])['total_sales'].sum().reset_index()
            fig = px.line(sales_over_time, x='timestamp', y='total_sales', color='product',
                          title='Sales Over Time by Product')
            st.plotly_chart(fig)

        elif selected == "Trends":
            st.title("Trends Page")
            trend_data = filtered_data.groupby(pd.Grouper(key='timestamp', freq='M')).agg(
                total_sales=('total_sales', 'sum'),
                units_sold=('units_sold', 'sum')
            ).reset_index()

            st.subheader("Total Sales Trend")
            fig_sales = px.line(trend_data, x='timestamp', y='total_sales', title="Total Sales Over Time")
            st.plotly_chart(fig_sales)

            st.subheader("Units Sold Trend")
            fig_units = px.line(trend_data, x='timestamp', y='units_sold', title="Units Sold Over Time")
            st.plotly_chart(fig_units)

        elif selected == "Product":
            st.title("Product Page")
            product_sales = filtered_data.groupby('product')['total_sales'].sum().reset_index()
            fig = px.bar(product_sales, x='product', y='total_sales', title='Total Sales by Product')
            st.plotly_chart(fig)

        elif selected == "Gender & Country Sales":
            st.title("Gender & Country Sales")
            gender_sales = filtered_data.groupby('gender')['total_sales'].sum().reset_index()
            country_sales = filtered_data.groupby('Country')['total_sales'].sum().reset_index()

            st.subheader("Sales by Gender")
            gender_fig = px.pie(gender_sales, names='gender', values='total_sales', title="Total Sales by Gender")
            st.plotly_chart(gender_fig)

            st.subheader("Sales by Country")
            country_fig = px.bar(country_sales, x='Country', y='total_sales', title="Total Sales by Country")
            st.plotly_chart(country_fig)


        elif selected == "Prediction":
            st.title("Sales Prediction")
            model = load_model()

            if model is not None:
                st.subheader("Enter Details for Prediction")
                units_sold = st.number_input("Units Sold", min_value=0, value=10)
                price_per_unit = st.number_input("Price per Unit", min_value=0.0, value=50.0)

                if st.button("Predict Total Sales"):
                    if units_sold > 0 and price_per_unit > 0:
                        try:
                            # Prepare the input data for prediction
                            input_data = pd.DataFrame([[units_sold, price_per_unit]], columns=['units_sold', 'price_per_unit'])
                            predicted_sales = model.predict(input_data)[0]
                            st.success(f"Predicted Total Sales: ${predicted_sales:,.2f}")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
            else:
                st.error("Model not loaded. Please check the model file.")
                # Download button for filtered data
                st.download_button(
                    label="Download Filtered Data",
                    data=filtered_data.to_csv(index=False),
                    file_name='filtered_sales_data.csv',
                    mime='text/csv',
                )
