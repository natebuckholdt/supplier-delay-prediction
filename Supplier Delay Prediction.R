############################################################
# Supplier Delay Prediction
# Linear Regression and Logistic Regression
############################################################

############################################################
# 1. Packages
############################################################

# install.packages("tidyverse")
# install.packages("caret")
library(tidyverse)
library(caret)

set.seed(123)

############################################################
# 2. Load and Inspect the Dataset
############################################################

supplier <- read.csv("supplier_purchase_orders_case3.csv")

head(supplier)
dim(supplier)
nrow(supplier)
ncol(supplier)
names(supplier)
str(supplier)
summary(supplier)

# Missing values check
colSums(is.na(supplier))

############################################################
# 3. Target Variable Review
############################################################

# Regression target: Delay_Days
# Classification target: Delay_Status

summary(supplier$Delay_Days)
table(supplier$Delay_Status)
prop.table(table(supplier$Delay_Status))

# Visualize regression target

ggplot(supplier, aes(x = Delay_Days)) +
  geom_histogram(bins = 40) +
  labs(
    title = "Distribution of Delay Days",
    x = "Delay Days",
    y = "Number of Purchase Orders"
  ) +
  theme_minimal()

# Visualize classification target

ggplot(supplier, aes(x = Delay_Status)) +
  geom_bar() +
  labs(
    title = "Distribution of Delay Status",
    x = "Delay Status",
    y = "Number of Purchase Orders"
  ) +
  theme_minimal()

############################################################
# 4. Create Clean Working Dataset
############################################################

# Important leakage note:
# For the linear regression model predicting Delay_Days,
# do not use Delay_Status or Actual_Delivery_Days.
# For the logistic regression model predicting Delay_Status,
# do not use Delay_Days or Actual_Delivery_Days.
# PO_ID and Supplier_ID are removed because they are identifiers.

supplier_work <- supplier %>%
  select(
    Supplier_Region,
    Supplier_Rating,
    Product_Category,
    Material_Criticality,
    Season,
    Order_Quantity,
    Unit_Cost,
    Order_Value,
    Lead_Time_Days,
    Distance_Miles,
    Expedited,
    Rush_Order,
    Supplier_Years,
    Past_Orders_Count,
    Prior_Delay_Rate,
    On_Time_Rate,
    Defect_Rate,
    Quality_Audit_Score,
    Order_Complexity,
    Delay_Days,
    Delay_Status
  ) %>%
  drop_na() %>%
  mutate(
    Supplier_Region = as.factor(Supplier_Region),
    Product_Category = as.factor(Product_Category),
    Material_Criticality = as.factor(Material_Criticality),
    Season = as.factor(Season),
    Expedited = as.factor(Expedited),
    Rush_Order = as.factor(Rush_Order),
    Delay_Status = as.factor(Delay_Status)
  )

head(supplier_work)
str(supplier_work)
dim(supplier_work)

############################################################
# 5. Predictive EDA
############################################################

# Summary table by delay status
supplier_work %>%
  group_by(Delay_Status) %>%
  summarise(
    n = n(),
    avg_delay_days = mean(Delay_Days),
    avg_supplier_rating = mean(Supplier_Rating),
    avg_order_quantity = mean(Order_Quantity),
    avg_order_value = mean(Order_Value),
    avg_lead_time_days = mean(Lead_Time_Days),
    avg_distance_miles = mean(Distance_Miles),
    avg_prior_delay_rate = mean(Prior_Delay_Rate),
    avg_on_time_rate = mean(On_Time_Rate),
    avg_defect_rate = mean(Defect_Rate),
    avg_quality_audit_score = mean(Quality_Audit_Score),
    avg_order_complexity = mean(Order_Complexity)
  )

# Delay days vs prior delay rate
ggplot(supplier_work, aes(x = Prior_Delay_Rate, y = Delay_Days)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Delay Days vs Prior Delay Rate",
    x = "Prior Delay Rate",
    y = "Delay Days"
  ) +
  theme_minimal()

# Delay days vs supplier rating
ggplot(supplier_work, aes(x = Supplier_Rating, y = Delay_Days)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Delay Days vs Supplier Rating",
    x = "Supplier Rating",
    y = "Delay Days"
  ) +
  theme_minimal()

# Delay status by expedited status
ggplot(supplier_work, aes(x = Expedited, fill = Delay_Status)) +
  geom_bar(position = "fill") +
  labs(
    title = "Delay Status by Expedited Status",
    x = "Expedited",
    y = "Proportion",
    fill = "Delay Status"
  ) +
  theme_minimal()

# Delay status by material criticality
ggplot(supplier_work, aes(x = Material_Criticality, fill = Delay_Status)) +
  geom_bar(position = "fill") +
  labs(
    title = "Delay Status by Material Criticality",
    x = "Material Criticality",
    y = "Proportion",
    fill = "Delay Status"
  ) +
  theme_minimal()

############################################################
# 6. Linear Regression: Predict Delay_Days
############################################################

# Build a linear regression model to predict Delay_Days.
# Leakage variables excluded: Delay_Status and Actual_Delivery_Days.

supplier_lm <- supplier_work %>%
  select(-Delay_Status)

# Train/test split using 70/30 ratio
train_index_lm <- createDataPartition(
  supplier_lm$Delay_Days,
  p = 0.70,
  list = FALSE
)

train_data_lm <- supplier_lm[train_index_lm, ]
test_data_lm  <- supplier_lm[-train_index_lm, ]

nrow(train_data_lm)
nrow(test_data_lm)

# Multiple linear regression model
model_delay_days <- lm(
  Delay_Days ~ Supplier_Region + Supplier_Rating + Product_Category +
    Material_Criticality + Season + Order_Quantity + Unit_Cost +
    Order_Value + Lead_Time_Days + Distance_Miles + Expedited +
    Rush_Order + Supplier_Years + Past_Orders_Count + Prior_Delay_Rate +
    On_Time_Rate + Defect_Rate + Quality_Audit_Score + Order_Complexity,
  data = train_data_lm
)

summary(model_delay_days)
coef(model_delay_days)

# Predict Delay_Days for test data
test_data_lm <- test_data_lm %>%
  mutate(
    pred_delay_days = predict(model_delay_days, newdata = test_data_lm),
    residual_delay_days = Delay_Days - pred_delay_days
  )

head(test_data_lm %>% select(Delay_Days, pred_delay_days, residual_delay_days), 20)

# Evaluation metrics
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

RMSE <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

metrics_lm <- data.frame(
  Model = c("Linear Regression"),
  MAE = c(MAE(test_data_lm$Delay_Days, test_data_lm$pred_delay_days)),
  RMSE = c(RMSE(test_data_lm$Delay_Days, test_data_lm$pred_delay_days))
)

metrics_lm

# Predicted vs actual plot
ggplot(test_data_lm, aes(x = pred_delay_days, y = Delay_Days)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0) +
  labs(
    title = "Predicted vs Actual Delay Days",
    x = "Predicted Delay Days",
    y = "Actual Delay Days"
  ) +
  theme_minimal()

# Residual plot
ggplot(test_data_lm, aes(x = pred_delay_days, y = residual_delay_days)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0) +
  labs(
    title = "Residual Plot for Linear Regression Model",
    x = "Predicted Delay Days",
    y = "Residual"
  ) +
  theme_minimal()

############################################################
# 7. Logistic Regression: Predict Delay_Status
############################################################

# Build a logistic regression model to predict Delay_Status.
# Leakage variables excluded: Delay_Days and Actual_Delivery_Days.

supplier_logit <- supplier_work %>%
  select(-Delay_Days)

# Make sure On_Time is the reference class and Delayed is the event
supplier_logit$Delay_Status <- factor(
  supplier_logit$Delay_Status,
  levels = c("On_Time", "Delayed")
)

levels(supplier_logit$Delay_Status)

table(supplier_logit$Delay_Status)
prop.table(table(supplier_logit$Delay_Status))

# Train/test split using 70/30 ratio
train_index_logit <- createDataPartition(
  supplier_logit$Delay_Status,
  p = 0.70,
  list = FALSE
)

train_data_logit <- supplier_logit[train_index_logit, ]
test_data_logit  <- supplier_logit[-train_index_logit, ]

prop.table(table(train_data_logit$Delay_Status))
prop.table(table(test_data_logit$Delay_Status))

# Logistic regression model
logit_delay_status <- glm(
  Delay_Status ~ Supplier_Region + Supplier_Rating + Product_Category +
    Material_Criticality + Season + Order_Quantity + Unit_Cost +
    Order_Value + Lead_Time_Days + Distance_Miles + Expedited +
    Rush_Order + Supplier_Years + Past_Orders_Count + Prior_Delay_Rate +
    On_Time_Rate + Defect_Rate + Quality_Audit_Score + Order_Complexity,
  data = train_data_logit,
  family = "binomial"
)

summary(logit_delay_status)

# Predicted probabilities for test data
test_data_logit <- test_data_logit %>%
  mutate(
    prob_delayed = predict(logit_delay_status, newdata = test_data_logit, type = "response")
  )

head(test_data_logit %>% select(Delay_Status, prob_delayed), 25)

# Histogram of predicted probabilities
ggplot(data = test_data_logit, aes(x = prob_delayed, fill = Delay_Status)) +
  geom_histogram(bins = 30, alpha = 0.5, position = "identity") +
  labs(
    title = "Predicted Probability of Delay",
    x = "Predicted Probability of Delayed",
    y = "Number of Purchase Orders",
    fill = "Actual Delay Status"
  ) +
  theme_minimal()

# Convert probabilities to predicted classes using 0.50 threshold
test_data_logit <- test_data_logit %>%
  mutate(
    pred_class = ifelse(prob_delayed >= .5, "Delayed", "On_Time"),
    pred_class = factor(pred_class, levels = levels(Delay_Status))
  )

head(test_data_logit %>% select(Delay_Status, prob_delayed, pred_class), 20)

# Accuracy
logit_accuracy <- mean(test_data_logit$pred_class == test_data_logit$Delay_Status)
logit_accuracy

# Confusion matrix preview using table
confusion_table <- table(
  Actual = test_data_logit$Delay_Status,
  Predicted = test_data_logit$pred_class
)
confusion_table

# Baseline accuracy: predict every purchase order as the majority class
baseline_accuracy <- max(prop.table(table(test_data_logit$Delay_Status)))
baseline_accuracy

# Does logistic regression improve over the baseline?
logit_accuracy
baseline_accuracy
logit_accuracy - baseline_accuracy

############################################################
# 8. Predictions for Existing Test Purchase Orders
############################################################

# How many days late or early is a purchase order likely to arrive?
head(test_data_lm %>% select(Delay_Days, pred_delay_days, residual_delay_days), 20)

# Is a purchase order likely to be delayed or on time?
head(test_data_logit %>% select(Delay_Status, prob_delayed, pred_class), 20)

############################################################
# 9. Save Prediction Outputs
############################################################

write.csv(test_data_lm, "supplier_case3_linear_regression_predictions.csv", row.names = FALSE)
write.csv(test_data_logit, "supplier_case3_logistic_regression_predictions.csv", row.names = FALSE)
