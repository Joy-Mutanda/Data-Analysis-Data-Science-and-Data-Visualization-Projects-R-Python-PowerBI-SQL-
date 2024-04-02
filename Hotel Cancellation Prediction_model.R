

library(readr)
library(caret)
library(tidyverse)
library(skimr)
library(readr)
library(ggplot2)
library(dplyr)
#library(moment)


#read in data
Hospitality_data <- read_csv("Hospitality_data3.csv")

#step 0 perform EDA
summaryStats <- skim(Hospitality_data)
summaryStats
str(Hospitality_data)

table(Hospitality_data$is_canceled)

# Dropping "reservation_status","reservation_status_date and "company" columns
Hospitality_data <- Hospitality_data[, !(names(Hospitality_data) %in% c("agent","company", "country","reservation_status","reservation_status_date"))]


# Create boxplots for each lead_time variable group by response variable
boxplot(Hospitality_data$lead_time~Hospitality_data$is_canceled)

# Create boxplots for each numeric variable in the dataset to see outliers
boxplot(Hospitality_data[, sapply(Hospitality_data, is.numeric)])

#step1 Partition the data and preprocesing
#1 change response to a factor
Hospitality_data$is_canceled<-as.factor(Hospitality_data$is_canceled)

Hospitality_data$is_canceled<-relevel(Hospitality_data$is_canceled, ref= "canceled")
levels(Hospitality_data$is_canceled)

Hospitality_data$assigned_room_type[Hospitality_data$assigned_room_type %in% c("B","C","E","F","G","H","I","K","L","P")] <- "Others"
Hospitality_data$reserved_room_type[Hospitality_data$reserved_room_type %in% c("B","C","E","F","G","H","L","P")] <- "Others"
Hospitality_data$market_segment[Hospitality_data$market_segment %in% c("Complementary","Aviation","Undefined")] <- "Others"
Hospitality_data$distribution_channel[Hospitality_data$distribution_channel %in% c("GDS","Undefined")] <- "Others"

#convert categorical variable
Hospitality_predictors_dummy<- model.matrix(is_canceled~.,
                                            data=Hospitality_data)

Hospitality_predictors_dummy<-data.frame(Hospitality_predictors_dummy[,-1])

Hospitality_data <-cbind(is_canceled=Hospitality_data$is_canceled, Hospitality_predictors_dummy)


#Data Partioning into Train and Test
library(caret)
set.seed(99) #set random seed
index <- createDataPartition(Hospitality_data$is_canceled, p = .8,list = FALSE)
Hospitality_train <-Hospitality_data[index,]
Hospitality_test <- Hospitality_data[-index,]


#install.packages("xgboost")
library(xgboost)

set.seed(8)
Hospitality_model <- train(is_canceled ~ .,
                           data = Hospitality_train,
                           method = "xgbTree",
                           trControl =trainControl(method = "cv", 
                                                   number = 5),
                           # provide a grid of parameters
                           tuneGrid = expand.grid(
                             nrounds = c(200, 300),
                             eta = c(0.05, 1),
                             max_depth = c(6, 8),
                             gamma = 10, #to prevent overfitting
                             colsample_bytree = 0.8,
                             min_child_weight = 1,
                             subsample = 0.8),
                           verbose=FALSE)
plot(Hospitality_model)
Hospitality_model$bestTune
plot(varImp(Hospitality_model))


xgboost_predprob<-predict(Hospitality_model, Hospitality_test, type="prob")
library(ROCR)
pred_xgboost <- prediction(xgboost_predprob$canceled, Hospitality_test$is_canceled,
                           label.ordering =c("notcanceled","canceled"))
#ROC Plot
perf_xgboost <- performance(pred_xgboost, "tpr", "fpr")
plot(perf_xgboost, colorize=TRUE)

#Get the AUC
auc<-unlist(slot(performance(pred_xgboost, "auc"), "y.values"))
auc

