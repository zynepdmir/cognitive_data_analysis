library(missMethods)
library(psych)
library(ggplot2)
library(dplyr)
library(PerformanceAnalytics)
library(GGally)
library(naniar)
library(mice)
library(lattice)
library(VIM)
library(corrplot)
library(vtreat)
library(car)
library(glmnet)
#install.packages('kerastuneR')
#kerastuneR::install_kerastuner()
library(keras)
library(tensorflow)
library(kerastuneR) 
library(xgboost)
library(caret)
library(randomForest)
library(doParallel)
library(e1071)


data <- read.csv("C:/Users/zynep/OneDrive/Desktop/stat412/project/human_cognitive_performance.csv")

#### data manipulation ####
data <- subset(data, select = -c(User_ID, AI_Predicted_Score))

cat_vars <- c("Gender", "Diet_Type", "Exercise_Frequency")
data[cat_vars] <- lapply(data[cat_vars], as.factor)

num_vars <- c("Age", "Sleep_Duration", "Stress_Level", "Daily_Screen_Time",
              "Caffeine_Intake", "Reaction_Time", "Memory_Test_Score", "Cognitive_Score")


data[num_vars] <- lapply(data[num_vars], as.numeric)


data$Stress_Level <- cut(data$Stress_Level,
                         breaks = c(0, 3, 7, 10),
                         labels = c("Low", "Medium", "High"),
                         right = TRUE,
                         include.lowest = TRUE)

data$Exercise_Frequency <- factor(data$Exercise_Frequency, 
                                     levels = c("Low", "Medium", "High"),  
                                     ordered = TRUE)

colnames(data) <- c("Age", "Gen", "SD", "SL", 
                    "DT", "DST", "EF", "CI", "RT", "MTS", "CS")

#### add missingness ####
# the data originally has no NA's so I add 10% of missingness 
# into the variables Reaction_Time and Caffeine_Intake
selected_vars = c("RT", "CI")
data = delete_MCAR(ds = data, p = 0.05, cols_mis = selected_vars)
sum(is.na(data))/nrow(data)

summary(data)
str(data)

#### exploratory data anaysis ####
# research question 1: Are people who don't exercise much have less cognitive score than others?
data %>%
  ggplot(aes(x = EF, y = CS)) +
  geom_boxplot(color =  "black",position = position_dodge(0.8)) +
  labs(
    title = "Cognitive Score by Exercise",
    x = "Exercise",
    y = "Cognitive Score"
  ) +
  theme_minimal()

# According to the box-plot of cognitive score among 3 exercise groups, low exercise group has
# low cognitive score whereas high exercise group have the highest cognitive score.
# And the medium exercise group's cognitive score's are between the two exercise group's scores.

# Research question 2: Is Cognitive Score differs between different genders?

ggplot(data, aes(x = Gen, y = CS, fill = Gen)) +
  geom_violin(trim = FALSE) +
  scale_fill_manual(values = c("Female" = "#E76449", "Male" = "#1C6AA8", "Other" = "#FFFF37")) +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(title = "Cognitive Score by Gender",
       x = "Gender",
       y = "Cognitive Score") +
  theme_minimal()

# Males and females doesn't seem to differ by cognitive score, both are right skewed.
# On the other hand, other genders' distribution is slightly different than male and females' 
# distributions. Other gender's distribution is more varied. 

# research question 3: how is cognitive score distributed?
data %>%
  ggplot(aes(x = CS)) + 
  geom_histogram( aes(y = after_stat(density)), binwidth = 3, color = "black", fill = "white") + 
  geom_density(alpha = 0.2, fill = "darkblue") +
  labs(
    title = "Distribution of Cognitive Score",
    x = "Cognitive Score",
    y = "Density"
  ) +
  theme_minimal()

# cognitive score's distribution is left skewed, might have some outliers on 100 scores.


# scatter plot matrix
set.seed(412)
sample_inx <- sample(1:nrow(data), 500)
data_subset <- data[sample_inx, ]

data_subset <- subset(data_subset, select = -c(Gen, DT, SL, EF))

data_subset$MF <- ifelse(rowSums(is.na(data_subset[, c("RT", "CI")])) > 0, "Missing", "Complete")


# plot
ggpairs(
  data_subset,
  columns = c("Age", "SD", "MTS","CS"),
  aes(color = MF)
)


#### missing value analysis ####
data %>% vis_miss()

data %>%
  bind_shadow(only_miss = TRUE) %>%
  group_by(CI_NA) %>% 
  summarize(score_mean = mean(CS), 
            score_sd = sd(CS))

gg_miss_upset(data)

# It appears to be the missing value structure is MCAR since the visual patterns are random
# and not showing any obvious patterns. There's 5% missing value in Caffeine_Intake and Reaction_Time,
# seperately.

md.pattern(data)
# We can see the missingness pattern in this table output.
# there are 3800 cases where Reaction_Time is missing
# and all other values are present. Similarly, there are 3800 cases
# where Caffeine_Intake is missing, and 200 cases where both 
# Reaction_Time and Caffeine_Intake is missing. The rest of the cases,
# 72200 values are present for all of the variables. 

#formal test
mcar_test(data)
# p-value is 0.446
# fail to reject the null hypothesis that the missingness structure is MCAR


data_miss = aggr(data, col=mdc(1:2),
                   numbers=TRUE, sortVars=TRUE,
                   labels=names(data),
                    ylab=c("Proportion of missingness","Missingness Pattern"))

mice_imputes = mice(data, m=5, maxit = 40)

mice_imputes$method
# mice used predictive mean matching method because both variables are numeric
densityplot(mice_imputes)
# the missingness structure is MAR since the pink imputed values are similar to the 
# blue imputed values.

imputed_data <- complete(mice_imputes,5)

# correlation matrix of numerical variables
num_vars <- imputed_data[, sapply(imputed_data, is.numeric)]
corr_matrix <- cor(num_vars)

corrplot(corr_matrix, method = "color",
         type ="lower",
         addCoef.col = "black",  
         tl.col = "black",        
         number.cex = 0.8)


#### confirmatory data analysis ####
# 1. test if exercise frequency affects the mean outcome
kruskal.test(CS ~ EF, data = imputed_data) # EF groups make difference
pairwise.wilcox.test(data$CS, data$EF, p.adjust.method = "bonferroni") 
# all groups's mean outcome is different than each other

data1 <- imputed_data
data1$LowEF <- ifelse(imputed_data$EF == "Low", "Low", "Other")
wilcox.test(CS ~ LowEF, data = data1)


leveneTest(CS ~ EF, data = imputed_data) 
# homogenity of variances assumption is satisfied

# 2. distribution of the cognitive score
library(nortest)
library(moments)
ad.test(imputed_data$CS)
skewness(imputed_data$CS)

# 3. multicolinearity check
model <- lm(CS ~ Age + SD + DST + RT + CI + MTS, data = imputed_data)
vif(model)
# there is no multicollinearity problem.


#### one-hot encoding for categorical variables ####
vars_to_encode <- c("Gen","SL", "DT", "EF")
treatment_plan <- designTreatmentsZ(imputed_data, varlist = vars_to_encode)
treated_data <- prepare(treatment_plan, imputed_data)

treated_data <- cbind(imputed_data, treated_data)
treated_data <- subset(treated_data, select = -c(SL_catP, Gen_catP, DT_catP, EF_catP))

#### cross validation ####
set.seed(412)  
train_index <- createDataPartition(treated_data$CS, p = 0.8, list = FALSE)

train_set <- treated_data[train_index, ]
test_set <- treated_data[-train_index, ]

#### Regression ####

train_set1 <- subset(train_set, 
                     select = -c(Gen_lev_x_Female, Gen_lev_x_Male, Gen_lev_x_Other,
                                 DT_lev_x_Non_minus_Vegetarian,
                                 DT_lev_x_Vegan, DT_lev_x_Vegetarian,
                                 EF_lev_x_High, EF_lev_x_Low, EF_lev_x_Medium,
                                 SL_lev_x_High, SL_lev_x_Medium, SL_lev_x_Low))

test_set1 <- subset(test_set,
                    select = -c(Gen_lev_x_Female, Gen_lev_x_Male, Gen_lev_x_Other,
                                DT_lev_x_Non_minus_Vegetarian,
                                DT_lev_x_Vegan, DT_lev_x_Vegetarian,
                                EF_lev_x_High, EF_lev_x_Low, EF_lev_x_Medium,
                                SL_lev_x_High, SL_lev_x_Medium, SL_lev_x_Low))

##### regression model 1 #####
lm.fit1 <- glm(CS~., data=train_set1)
summary(lm.fit1)

# test performance for model 1
pr.test1 <- predict(lm.fit1,test_set1)
MSE.test1 <- sum((pr.test1 - test_set1$CS)^2)/nrow(test_set1)
RMSE.test1 <- sqrt(mean((test_set1$CS - pr.test1)^2))
MAE.test1 <- mean(abs(test_set1$CS - pr.test1))
rss_test1 <- sum((test_set1$CS - pr.test1)^2)
tss_test1 <- sum((test_set1$CS - mean(test_set1$CS))^2)
r_squared.test1 <- 1 - rss_test1/tss_test1

# train performance for model 1
train_pred1 <- predict(lm.fit1, train_set1)
MSE.train1 <- mean((train_set1$CS - train_pred1)^2)
RMSE.train1 <- sqrt(MSE.train1)
MAE.train1 <- mean(abs(train_set1$CS - train_pred1))
rss.train1 <- sum((train_set1$CS - train_pred1)^2)
tss.train1 <- sum((train_set1$CS - mean(train_set1$CS))^2)
r_squared.train1 <- 1 - rss.train1 / tss.train1

# checking multicolinearity
vif(lm.fit1) # there's no multicolinearity

##### regression model 2:try interactions #####
lm.fit2 <- glm(CS ~ SD * EF + RT + MTS + SL + DST + CI, data = train_set1)
summary(lm.fit2)

# test performance for model 2
pr.test2 <- predict(lm.fit2,test_set1)
MSE.test2 <- sum((pr.test2 - test_set1$CS)^2)/nrow(test_set1)
RMSE.test2 <- sqrt(mean((test_set1$CS - pr.test2)^2))
MAE.test2 <- mean(abs(test_set1$CS - pr.test2))
rss_test2 <- sum((test_set1$CS - pr.test2)^2)
tss_test2 <- sum((test_set1$CS - mean(test_set1$CS))^2)
r_squared.test2 <- 1 - rss_test2/tss_test2

# train performance for model 2
train_pred2 <- predict(lm.fit2, train_set1)
MSE.train2 <- mean((train_set1$CS - train_pred2)^2)
RMSE.train2 <- sqrt(MSE.train2)
MAE.train2 <- mean(abs(train_set1$CS - train_pred2))
rss.train2 <- sum((train_set1$CS - train_pred2)^2)
tss.train2 <- sum((train_set1$CS - mean(train_set1$CS))^2)
r_squared.train2 <- 1 - rss.train2 / tss.train2

##### regression model 3 #####
step_fit <- step(lm.fit1, direction = "both")
summary(step_fit)

# test performance for model 3
pr.test3 <- predict(step_fit, test_set1)
MSE.test3 <- sum((pr.test3 - test_set1$CS)^2)/nrow(test_set1)
RMSE.test3 <- sqrt(mean((test_set1$CS - pr.test3)^2))
MAE.test3 <- mean(abs(test_set1$CS - pr.test3))
rss_test3 <- sum((test_set1$CS - pr.test3)^2)
tss_test3 <- sum((test_set1$CS - mean(test_set1$CS))^2)
r_squared.lm3 <- 1 - rss_test3/tss_test3

# train performance for model 3
train_pred3 <- predict(step_fit, train_set1)
MSE.train3 <- mean((train_set1$CS - train_pred3)^2)
RMSE.train3 <- sqrt(MSE.train3)
MAE.train3 <- mean(abs(train_set1$CS - train_pred3))
rss.train3 <- sum((train_set1$CS - train_pred3)^2)
tss.train3 <- sum((train_set1$CS - mean(train_set1$CS))^2)
r_squared.train3 <- 1 - rss.train3 / tss.train3

##### regression models 4 and 5 #####
x <- model.matrix(CS ~ . -1, data = train_set1)
x_test <- model.matrix(CS ~ . -1, data = test_set1)
y <- train_set1$CS

cv_fit_lasso <- cv.glmnet(x, y, alpha = 1)  
cv_fit_ridge <- cv.glmnet(x, y, alpha = 0) 

# test performance for model 4
pr.test4 <- predict(cv_fit_lasso, newx = x_test, s = "lambda.min")
MSE.test4 <- sum((pr.test4 - test_set1$CS)^2)/nrow(test_set1)
RMSE.test4 <- sqrt(mean((test_set1$CS - pr.test4)^2))
MAE.test4 <- mean(abs(test_set1$CS - pr.test4))
rss_test4 <- sum((test_set1$CS - pr.test4)^2)
tss_test4 <- sum((test_set1$CS - mean(test_set1$CS))^2)
r_squared.test4 <- 1 - rss_test4/tss_test4

# train performance for model 4
X_train <- model.matrix(CS ~ . -1, data = train_set1)
pr.train4 <- predict(cv_fit_lasso, newx = X_train, s = "lambda.min")
MSE.train4 <- sum((pr.train4 - train_set1$CS)^2)/nrow(train_set1)
RMSE.train4 <- sqrt(MSE.train4)
MAE.train4 <- mean(abs(train_set1$CS - pr.train4))
rss.train4 <- sum((train_set1$CS - pr.train4)^2)
tss.train4 <- sum((train_set1$CS - mean(train_set1$CS))^2)
r_squared.train4 <- 1 - rss.train4 / tss.train4

# test performance for model 5
pr.test5 <- predict(cv_fit_ridge, newx = x_test, s = "lambda.min")
MSE.test5 <- sum((pr.test5 - test_set1$CS)^2)/nrow(test_set1)
RMSE.test5 <- sqrt(mean((test_set1$CS - pr.test5)^2))
MAE.test5 <- mean(abs(test_set1$CS - pr.test5))
rss_test5 <- sum((test_set1$CS - pr.test5)^2)
tss_test5 <- sum((test_set1$CS - mean(test_set1$CS))^2)
r_squared.test5 <- 1 - rss_test5/tss_test5

# train performance for model 5
pr.train5 <- predict(cv_fit_ridge, newx = X_train, s = "lambda.min")
MSE.train5 <- sum((pr.train5 - train_set1$CS)^2)/nrow(train_set1)
RMSE.train5 <- sqrt(MSE.train5)
MAE.train5 <- mean(abs(train_set1$CS - pr.train5))
rss.train5 <- sum((train_set1$CS - pr.train5)^2)
tss.train5 <- sum((train_set1$CS - mean(train_set1$CS))^2)
r_squared.train5 <- 1 - rss.train5 / tss.train5

##### comparison #####
results_df_reg <- data.frame(
  Model = paste0("Model", 1:5),
  
  MSE_Train = c(MSE.train1, MSE.train2, MSE.train3, MSE.train4, MSE.train5),
  RMSE_Train = c(RMSE.train1, RMSE.train2, RMSE.train3, RMSE.train4, RMSE.train5),
  MAE_Train  = c(MAE.train1,  MAE.train2,  MAE.train3,  MAE.train4,  MAE.train5),
  R2_Train   = c(r_squared.train1, r_squared.train2, r_squared.train3, r_squared.train4, r_squared.train5),
  
  MSE_Test = c(MSE.test1, MSE.test2, MSE.test3, MSE.test4, MSE.test5),
  RMSE_Test = c(RMSE.test1, RMSE.test2, RMSE.test3, RMSE.test4, RMSE.test5),
  MAE_Test  = c(MAE.test1,  MAE.test2,  MAE.test3,  MAE.test4,  MAE.test5),
  R2_Test   = c(r_squared.test1, r_squared.test2, r_squared.lm3, r_squared.test4, r_squared.test5)
)

print(results_df_reg)

best_rmse_test_model <- results_df[which.min(results_df$RMSE_Test), ] #2
best_r2_test_model <- results_df[which.max(results_df$R2_Test), ] #2
best_mse_test_model <- results_df[which.min(results_df$MSE_Test), ] #2
best_mae_test_model <- results_df[which.max(results_df$MAE_Test), ] #5

best_rmse_train_model <- results_df[which.min(results_df$RMSE_Train), ] #2
best_r2_train_model <- results_df[which.max(results_df$R2_Train), ] #2
best_mse_train_model <- results_df[which.min(results_df$MSE_Train), ] #2
best_mae_train_model <- results_df[which.max(results_df$MAE_Train), ] #5

model2_df <- data.frame(
  model = c("GLR", "GLR"),
  Set = c("Train", "Test"),
  MSE = c(MSE.train2, MSE.test2),
  RMSE = c(RMSE.train2, RMSE.test2),
  MAE = c(MAE.train2, MAE.test2),
  `R_squared` = c(r_squared.train2, r_squared.test2)
)

# the best regression model is model2, the model with interaction terms

##### visuals #####

# Predict on test or train set
predicted <- predict(lm.fit2, newdata = test_set1)
actual <- test_set1$CS

# Create data frame
plot_df <- data.frame(Actual = actual, Predicted = predicted)

# Plot
ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#1f77b4", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Predicted vs Actual Values",
    x = "Actual CS",
    y = "Predicted CS"
  )

residuals <- actual - predicted

ggplot(data.frame(Predicted = predicted, Residuals = residuals), aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.6, color = "#2ca02c") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    title = "Residuals vs Predicted",
    x = "Predicted CS",
    y = "Residuals"
  )

#### scaling ####
num_data <- subset(treated_data, select = c("Age", "SD", "DST",
                                            "CI", "RT", "MTS", "CS"))
maxs <- apply(num_data, 2, max) 
mins <- apply(num_data, 2, min)
scaled_data <- as.data.frame(scale(num_data, center = mins, scale = maxs - mins))
imputed_data <- cbind(scaled_data, subset(imputed_data, select = c("Gen", "SL","DT", "EF")))


#### ANN model ####
train_set2 <- subset(train_set, 
                     select = -c(SL, EF, DT, Gen))

test_set2 <- subset(test_set,
                    select = -c(SL, EF, DT, Gen))

X_train2 <- as.matrix(subset(train_set2, select = -c(CS)))
X_test2 <- as.matrix(subset(test_set2, select = -c(CS)))
y_train2 <-train_set2$CS
y_train2 <- matrix(y_train2, ncol = 1)
y_test2 <- test_set2$CS
y_test2 <- matrix(y_test2, ncol = 1)

##### Model building function #####
build_model <- function(hp) {
  model <- keras_model_sequential() %>%
    layer_dense(
      units = hp$Int("units", 16, 128, step = 16),
      activation = "relu",
      input_shape = ncol(X_train2))%>% 
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    optimizer = optimizer_adam(
      learning_rate = hp$Choice("lr", values = c(1e-2, 1e-3, 1e-4))),
    loss = "mse",
    metrics = list("mae"))
  return(model)
}

##### Tuner setup #####
tuner <- RandomSearch(
  hypermodel = build_model,
  objective = "val_mae",
  max_trials = 10,
  executions_per_trial = 1,
  overwrite = TRUE
)

##### Search #####
tuner %>% fit_tuner(
  x = X_train2,
  y = y_train2,
  epochs = 10,
  validation_data = list(
    x = X_test2,
    y = matrix(y_test2, ncol = 1)  
  )
  ,
  verbose = 0
)

best_hps_all <- tuner$get_best_hyperparameters()

length(best_hps_all)

# Extract and format them into a data frame
best_hps_df <- bind_rows(lapply(best_hps_all[1:5], function(hp) {
  as.data.frame(hp$values, stringsAsFactors = FALSE)
}))
print(best_hps_df)

# units    lr
# 80 0.001

# use these 48 units and 0.001 learning rate hyperparameters in ANN model
model <- keras_model_sequential() %>%
  layer_dense(units = 48, activation = "relu", input_shape = ncol(X_train2)) %>%
  layer_dense(units = 1, activation = "linear")

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse",
  metrics = list("mae")
)

history <- keras::fit(
  object = model,
  x = X_train2,
  y = y_train2,
  epochs = 50,
  validation_data = list(X_test2, y_test2),
  verbose = 1
)

##### Predictions #####
pred_train_ANN <- model %>% predict(X_train2)
pred_test_ANN <- model %>% predict(X_test2)

##### Metrics #####
MSE_train_ANN <- mean((y_train2 - pred_train_ANN)^2)
RMSE_train_ANN <- sqrt(MSE_train_ANN)
MAE_train_ANN <- mean(abs(y_train2 - pred_train_ANN))
R2_train_ANN <- 1 - sum((y_train2- pred_train_ANN)^2) / sum((y_train2 - mean(y_train2))^2)

MSE_test_ANN <- mean((y_test2 - pred_test_ANN)^2)
RMSE_test_ANN <- sqrt(MSE_test_ANN)
MAE_test_ANN <- mean(abs(y_test2 - pred_test_ANN))
R2_test_ANN <- 1 - sum((y_test2 - pred_test_ANN)^2) / sum((y_test2 - mean(y_test2))^2)

results_df_ANN <- data.frame(
  model = c("ANN", "ANN"),
  Set = c("Train", "Test"),
  MSE = c(MSE_train_ANN, MSE_test_ANN),
  RMSE = c(RMSE_train_ANN, RMSE_test_ANN),
  MAE = c(MAE_train_ANN, MAE_test_ANN),
  R_squared = c(R2_train_ANN, R2_test_ANN)
)

print(results_df_ANN)


library(forecast)
BoxCox.lambda(data$CS)
library(nortest)
ad.test(scaled_data$CS)
data2 <- data
data2$CS <- BoxCox(data2$CS, BoxCox.lambda(data2$CS))
ad.test(data2$CS)

 #### XGBoost Model ####

# prepare data
X_train_matrix <- as.matrix(subset(train_set2, select = -CS))
y_train_vector <- train_set2$CS

dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train_vector)

# set parameter grid 
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

##### cross validation #####
set.seed(412)
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 100,
  nfold = 5,
  metrics = "rmse",
  early_stopping_rounds = 10,
  verbose = 1
)

final_model <- xgboost(
  data = dtrain,
  params = params,
  nrounds = cv_model$best_iteration,
  verbose = 0
)

# Prepare test data
X_test_matrix <- model.matrix(CS ~ . -1, data = test_set2)
y_test_vector <- test_set2$CS

##### Predictions #####
pred_train <- predict(final_model, newdata = X_train_matrix)
pred_test <- predict(final_model, newdata = X_test_matrix)

# Train metrics
mse_train <- mean((y_train_vector - pred_train)^2)
rmse_train <- sqrt(mse_train)
mae_train <- mean(abs(y_train_vector - pred_train))
r2_train <- 1 - sum((y_train_vector - pred_train)^2) / sum((y_train_vector - mean(y_train_vector))^2)

# Test metrics
mse_test <- mean((y_test_vector - pred_test)^2)
rmse_test <- sqrt(mse_test)
mae_test <- mean(abs(y_test_vector - pred_test))
r2_test <- 1 - sum((y_test_vector - pred_test)^2) / sum((y_test_vector - mean(y_test_vector))^2)

##### Summary table #####
results_df_xgb <- data.frame(
  model = c("XGB", "XGB"),
  Set = c("Train", "Test"),
  MSE = c(mse_train, mse_test),
  RMSE = c(rmse_train, rmse_test),
  MAE = c(mae_train, mae_test),
  R_squared = c(r2_train, r2_test)
)

print(results_df_xgb)

#visuals
xgb.importance(model = final_model) %>%
  xgb.plot.importance(top_n = 10)
# reaction has the most importance in the XGBoost model

#### SVM model ####
##### prepare the data #####
# sampling because svm do not work with large data
set.seed(412)
sample_index <- sample(1:nrow(treated_data), 10000)
treated_data2 <- treated_data[sample_index, ]

set.seed(412)  
train_index2 <- createDataPartition(treated_data2$CS, p = 0.8, list = FALSE)

train_set3 <- treated_data2[train_index2, ]
test_set3 <- treated_data2[-train_index2, ]

train_set3 <- subset(train_set3, 
                     select = -c(SL, EF, DT, Gen))

test_set3 <- subset(test_set3,
                    select = -c(SL, EF, DT, Gen))

X_train_svm <- model.matrix(CS ~ . -1, data = train_set3)  
y_train_svm <- train_set3$CS

X_test_svm <- model.matrix(CS ~ . -1, data = test_set3)
y_test_svm <- test_set3$CS

train_df <- data.frame(CS = y_train_svm, X_train_svm)

# hyperparameter tuning for cost and epsilon parameters
ctrl_svm <- trainControl(method = "cv", number = 5)
tuneGrid = data.frame(C = c(0.1, 1, 10, 100))

# parallel processing (because svm took so much time to run)
cl <- makePSOCKcluster(4)  
registerDoParallel(cl)

##### train SVM model #####
set.seed(412)
svm_model <- train(
  CS ~ .,
  data = train_set3,
  method = "svmLinear",
  trControl = ctrl_svm,
  tuneGrid = data.frame(C = c(1, 10)),
  metric = "RMSE"
)

# Stop parallel processing
stopCluster(cl)

print(svm_model)
svm_model$bestTune

##### Predict on test set ####
pred_test <- predict(svm_model, newdata = test_set3)
pred_train <- predict(svm_model, newdata = train_set3)

pred_test <- pmin(pmax(pred_test, 0), 100)
pred_train <- pmin(pmax(pred_train, 0), 100)

##### Evaluate performance #####
results_df_svm <- data.frame(
  model = c("SVM", "SVM"),
  Set = c("Train", "Test"),
  MSE = c(mean((train_set3$CS - pred_train)^2), mean((test_set3$CS - pred_test)^2)),
  RMSE = c(sqrt(mean((train_set3$CS - pred_train)^2)), sqrt(mean((test_set3$CS - pred_test)^2))),
  MAE = c(mean(abs(train_set3$CS - pred_train)), mean(abs(test_set3$CS - pred_test))),
  R_squared = c(
    1 - sum((train_set3$CS - pred_train)^2) / sum((train_set3$CS - mean(train_set3$CS))^2),
    1 - sum((test_set3$CS - pred_test)^2) / sum((test_set3$CS - mean(test_set3$CS))^2)
  )
)
print(results_df_svm)

##### Plot #####
plot_df <- data.frame(Actual = test_set3$CS, Predicted = pred_test)
sample_idx <- sample(1:nrow(plot_df), size = min(500, nrow(plot_df)))


ggplot(plot_df[sample_idx, ], aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(title = "SVM Predictions vs Actual (Test Set)", x = "Actual CS", y = "Predicted CS")

#### Random Forest model ####
##### prepare the datasets #####
# I will sample again because the original dataset training took so much time
# I will also apply parallel processing
set.seed(412)
sample_index <- sample(1:nrow(treated_data), 10000)
treated_data2 <- treated_data[sample_index, ]
train_index2 <- createDataPartition(treated_data2$CS, p = 0.8, list = FALSE)

train_set3 <- treated_data2[train_index2, ]
test_set3 <- treated_data2[-train_index2, ]

train_set3 <- subset(train_set3, 
                     select = -c(SL, EF, DT, Gen))

test_set3 <- subset(test_set3,
                    select = -c(SL, EF, DT, Gen))

X_train <- model.matrix(CS ~ . -1, data = train_set3)
y_train <- train_set3$CS

X_test <- model.matrix(CS ~ . -1, data = test_set3)
y_test <- test_set3$CS

train_data <- data.frame(CS = y_train, X_train)
test_data <- data.frame(CS = y_test, X_test)

##### cross validation #####
ctrl_rf <- trainControl(method = "cv", number = 2)

# Grid for mtry 
mtry_grid <- expand.grid(mtry = c(5, 10, 15))

# Parallel backend
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

##### Train the model #####
set.seed(412)
rf_model <- caret::train(
  CS ~ .,
  data = train_data,
  method = "ranger",
  trControl = ctrl_rf,
  num.trees = 50,
  importance = "impurity",
  metric = "RMSE"
)


stopCluster(cl)

##### Predictions #####
pred_train <- predict(rf_model, newdata = train_data)
pred_test <- predict(rf_model, newdata = test_data)

# clip predictions
pred_test <- pmin(pmax(pred_test, 0), 100)

##### evaluate performance #####
results_df_rf <- data.frame(
  model = c("RF", "RF"),
  Set = c("Train", "Test"),
  MSE = c(mean((train_set3$CS - pred_train)^2), mean((test_set3$CS - pred_test)^2)),
  RMSE = c(sqrt(mean((train_set3$CS - pred_train)^2)), sqrt(mean((test_set3$CS - pred_test)^2))),
  MAE = c(mean(abs(train_set3$CS - pred_train)), mean(abs(test_set3$CS - pred_test))),
  R_squared = c(
    1 - sum((train_set3$CS - pred_train)^2) / sum((train_set3$CS - mean(train_set3$CS))^2),
    1 - sum((test_set3$CS - pred_test)^2) / sum((test_set3$CS - mean(test_set3$CS))^2)
  )
)
print(results_df_rf)

##### variable importance #####
importance_vals <- varImp(rf_model)
# random forest model's most important variable is Reaction_Time, too

##### plot #####
plot(importance_vals)

#### compare all models ####
all_models_df <- rbind(model2_df, results_df_ANN, results_df_rf, results_df_svm, results_df_xgb)

# best model wrt test MSE
test_set_all <- all_models_df[all_models_df$Set == "Test", ]
best_mse_test <- test_set_all[which.min(test_set_all$MSE), ]
best_mse_test

# best model wrt train MSE
train_set_all <- all_models_df[all_models_df$Set == "Train", ]
best_mse_train <- train_set_all[which.min(train_set_all$MSE), ]
best_mse_train

# best model wrt test R_squared
best_rsq_test <- test_set_all[which.max(test_set_all$R_squared), ]
best_rsq_test

# best model wrt train R_squared
best_rsq_train <- train_set_all[which.max(train_set_all$R_squared), ]
best_rsq_train

# random forest is the best model if we evaluate the train accuracy,
# but when we consider the test accuracy, artificial neural networks has the best 
# performance. 
