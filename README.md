# Supplier Delay Prediction

## Business Problem

Late supplier deliveries can disrupt production schedules, increase costs, create inventory shortages, and reduce customer service levels. This project uses historical purchase-order data to identify factors associated with supplier delays and predict whether an order is likely to arrive late.

## Objective

Develop analytical models that help managers:

1. Estimate the number of days an order may be late or early.
2. Identify purchase orders at risk of being delayed.

## Tools Used

* R
* Excel
* Regression analysis
* Logistic regression
* Data visualization

## Data Preparation

The dataset included purchase-order details such as supplier information, product category, order size, cost, lead time, shipping method, distance, and delivery outcomes.

To avoid data leakage, outcome-related fields were excluded from predictor variables when building the models.

## Analysis

* Conducted exploratory data analysis to identify delay patterns.
* Built a linear regression model to predict delay days.
* Built a logistic regression model to classify orders as delayed or on time.
* Evaluated results using model performance measures and visualizations.

## Key Findings

* Certain supplier, shipping, lead-time, and order characteristics were associated with higher delay risk.
* The classification model can help managers prioritize at-risk orders before delays occur.
* Results can support proactive supplier communication, expedited-shipping decisions, and inventory planning.

## Business Recommendation

Use the delay-risk model as an early-warning tool. Orders identified as high risk should receive additional monitoring, supplier follow-up, and contingency planning.

## Files

* `Supplier Delay Prediction.R` — R code for the analysis
* `report.pdf` — written business report
* `images/` — selected visualizations and outputs

## Author

Nate Buckholdt
MBA Candidate, Business Analytics
San Diego, CA
