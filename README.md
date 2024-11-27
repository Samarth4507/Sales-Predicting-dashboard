# **Sales Dashboard**

## **Overview**

The **Sales Dashboard** is an interactive web application that provides comprehensive insights into sales data. Built with **Streamlit**, **Plotly**, and machine learning, the dashboard allows users to analyze sales trends, compare product performance, and predict future sales.

---

## **Features**

1. **Dynamic Sales Insights**
   - Visualize total sales, units sold, and top-performing products and countries.
   - Interactive filters for products, dates, countries, and demographics.

2. **Trend Analysis**
   - Identify monthly and yearly sales trends.
   - Understand seasonality with time-series visualizations.

3. **Product and Demographic Analysis**
   - Break down sales performance by product, gender, and country.
   - Compare sales using pie charts and bar graphs.

4. **Sales Prediction**
   - Predict total sales using a trained machine learning model.
   - Visualize actual vs. predicted sales for deeper insights.

5. **Data Export**
   - Download filtered data for further offline analysis.

---

## **Technologies Used**

- **Frontend:** [Streamlit](https://streamlit.io)
- **Visualizations:** [Plotly](https://plotly.com)
- **Machine Learning:** scikit-learn, RandomForestRegressor
- **Backend Logic:** Python (Pandas, NumPy)

---
sales-dashboard/
│
├── Salesdashboard.py      # Main application script
├── sales_data.csv         # Example dataset
├── regression_model.pkl   # Trained machine learning model
├── README.md              # Documentation
# **Sales Dashboard**

## **Dataset Requirements**

Ensure your dataset contains the following columns:

- **timestamp**: Date and time of the sale (format: `YYYY-MM-DD HH:MM`).
- **customer_id**: Unique identifier for each customer.
- **gender**: Customer gender (e.g., `Male`, `Female`).
- **Country**: Country where the sale occurred.
- **product**: Name of the product sold.
- **units_sold**: Quantity of units sold.
- **price_per_unit**: Price of each unit.
- **total_sales**: Total revenue generated (`units_sold * price_per_unit`).

---

## **Usage**

### **Filters**
Use the sidebar to filter sales data by:
- **Products**: Select specific products.
- **Dates**: Specify a date range for analysis.
- **Countries**: Filter by one or multiple countries.
- **Gender**: Filter by gender.

### **Predictions**
1. Navigate to the **"Prediction"** section.
2. Enter the required inputs:
   - Units Sold
   - Price Per Unit
3. View:
   - Predicted total sales.
     
### **Sample Visualizations**
1. **Sales Trends**: Line chart of sales over time.
2. **Product Performance**: Bar chart of total sales by product.
3. **Demographics**: Pie chart of sales by gender or country.

---

## **Model Details**

The dashboard includes a **Random Forest Regressor** model trained to predict sales using the following:

- **Features**: `units_sold`, `price_per_unit`.
- **Target**: `total_sales`.

### **Model Evaluation Metrics**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**


