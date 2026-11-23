# Retail Customer Sales: Exploratory Data Analysis

**Repository:** [https://github.com/LM1264/EDA](https://github.com/LM1264/EDA)

**Mini Project 1 | Data Analyst Training Program | MCA, First Year**

| | |
|---|---|
| **Student** | Lakshay Malik |
| **Admission No.** | 2026M0141 0097 |
| **Instructor** | Mr. Amit Bhatnagar |
| **Date** | 23/11/2026 |
| **Repository** | [github.com/LM1264/EDA](https://github.com/LM1264/EDA) |
| **GitHub profile** | [github.com/LM1264](https://github.com/LM1264) |

---

## 1. Overview

This project is a complete Exploratory Data Analysis (EDA) of a retail sales dataset with **7,084 raw transactions and 24 columns** covering 2023 and 2024 across four regions of India. The raw data was deliberately messy. I audited it, cleaned it (with a written reason for every decision), described it statistically, checked for outliers, created new columns and answered 8 business questions. The result is a cleaned dataset of **7,000 rows** and a set of reports.

## 2. Objectives

- Audit data quality: missing values, duplicates, inconsistent labels, impossible values, mixed date formats.
- Clean the data and justify every imputation, removal or flag.
- Compute descriptive statistics for 8 key numerical variables.
- Analyse numerical and categorical variables one at a time (univariate analysis).
- Detect outliers with the IQR and Z-score methods and judge whether each group is genuine or an error.
- Create calculated columns and answer business questions with `groupby`.
- Write a short insights report for a non-technical manager.

## 3. Repository Contents

```
EDA/
├── README.md                              <- this file
├── README.pdf                             <- PDF copy of this file
├── Retail_EDA_Notebook.ipynb              <- main notebook (code, outputs, explanations)
├── Source_Code.py                         <- the notebook exported as a Python script
├── retail_customer_sales_dataset.csv      <- raw data (needed to run the notebook)
├── Cleaned_Dataset.csv                    <- cleaned data, 7,000 rows x 39 columns
├── Insights_Report.pdf                    <- 3-page report for the manager
├── Project_Report.pdf                     <- full project report (26 pages)
├── SRS.pdf                                <- Software Requirements Specification
├── Test_Cases.pdf                         <- 45 test cases and results
├── Synopsis.pdf                           <- project synopsis
├── Data_Dictionary_and_Cleaning_Log.pdf   <- column meanings and every cleaning decision
├── Notebook_Execution_Output.pdf          <- printed notebook with outputs
└── images/
    └── correlation_heatmap.png            <- bonus correlation heatmap
```

## 4. Dataset

| Group | Columns |
|---|---|
| Identifiers | Transaction_ID, Customer_ID |
| Dates | Order_Date, Delivery_Date |
| Customer | Age, Gender, Income, Customer_Segment |
| Location | Region, State, City |
| Product | Product_Category, Product |
| Transaction | Quantity, Unit_Price, Discount, Sales, Cost, Profit |
| Service | Payment_Method, Delivery_Mode, Customer_Rating, Satisfaction_Level, Returned |

The cleaned file keeps these 24 columns and adds **6 flag columns** (for example `Sales_Outlier`, `Rating_Missing`) and **9 engineered columns** (for example `Profit_Margin`, `Delivery_Days`, `Revenue_per_Customer`).

## 5. What Was Wrong With the Data, and What I Did

| Problem found | How many | Fix |
|---|---|---|
| Duplicate transactions (61 exact copies plus 23 hidden by different date or spelling styles) | 84 rows | Standardised first, then removed |
| Region written 12 ways, Gender 8 ways | all rows | Mapped to 4 regions and 2 genders |
| Dates in 4 formats | all rows | Each format parsed by its own rule so day and month never swap |
| Impossible ages (-12, 150, 180, 999) | 6 rows | Treated as missing, then filled from the same customer or the median |
| Invalid discounts (below 0 or above 50%) | 6 rows | Rebuilt exactly from `1 - Sales / (Quantity x Unit_Price)` |
| Delivery dated before the order | 15 rows | Date removed and flagged; the sale was kept |
| Income 10 to 15 times too high on one order of a customer | 9 customers | Replaced with the customer's usual income |
| Missing values (rating, income, satisfaction, age, delivery date, discount) | 1,179 cells | Handled column by column (recalculated, filled or left blank on purpose) |

The notebook ends its cleaning with `assert` checks (unique IDs, valid ranges, one income per customer, and so on) that stop the run if any rule is broken.

## 6. Key Findings

1. **A few big orders carry the business.** The median order is about ₹4,826 but the average is ₹10,962. Electronics and Furniture are 42% of orders and 83% of sales.
2. **Sales are flat.** 2024 was 1.5% below 2023.
3. **Margins are uniform** at about 32.8%, and every category is within 0.4 points.
4. **The delivery promise is not met.** Express (5.6 days), Standard (5.7) and Same-Day (5.8) take about the same time.
5. **One order in eight is returned** (12.6%), worth about 11.8% of sales.
6. **Discounts do not increase basket size.** Units per order stay near 3 in every discount band.
7. **Customers are loyal.** Only 5.5% bought once, and the top 20% of customers bring about 45% of revenue.

The differences between regions, segments and categories are small, so the clearest findings are the delivery, returns and discount results. The dataset appears to be simulated.

### Sample outputs

![Monthly sales](images/monthly_sales.png)

![Correlation heatmap](images/correlation_heatmap.png)

Sales, Cost and Profit are strongly correlated (Sales and Cost are 0.99). Age, Income, Discount, Delivery Days and Customer Rating show almost no correlation with anything else.

## 7. Technology

| Library | Used for |
|---|---|
| Python 3.10+ | Language (tested on 3.13) |
| pandas | Loading, cleaning, grouping |
| numpy | Numbers and Z-score maths |
| matplotlib | Base charts |
| seaborn | Distribution, box and heat map plots |
| Jupyter Notebook | Code, outputs and notes in one file |

Optional libraries from the brief (`missingno`, `openpyxl`) are not needed.

## 8. How to Run

```bash
# 1. Install the libraries
pip install pandas numpy matplotlib seaborn jupyter

# 2. Keep the raw CSV in the same folder as the notebook, then start Jupyter
jupyter notebook Retail_EDA_Notebook.ipynb

# 3. Run all cells from top to bottom (takes about 7 seconds).
#    The notebook creates a figures/ folder and an outputs/ folder, and writes Cleaned_Dataset.csv.
```

On **Google Colab**, upload `Retail_EDA_Notebook.ipynb` and `retail_customer_sales_dataset.csv`, then choose *Runtime > Run all*. The notebook works on pandas 2 and pandas 3.

## 9. Testing

45 test cases check loading, auditing, cleaning, statistics, outliers, new columns, grouping and the deliverables. All 45 passed. Details and the five defects found and fixed during development are in `Test_Cases.pdf`.

## 10. Future Work

- **Mini Project 2 (Data Visualization):** interactive dashboards on the cleaned dataset.
- **Mini Project 3 (Advanced Statistics):** hypothesis tests (for example, whether delivery mode really changes delivery time), correlation and regression.
- Predictive models for returns and customer value.

## 11. Author

**Lakshay Malik**, MCA First Year, Admission No. 2026M0141 0097  
GitHub: [https://github.com/LM1264](https://github.com/LM1264)  
Repository: [https://github.com/LM1264/EDA](https://github.com/LM1264/EDA)
