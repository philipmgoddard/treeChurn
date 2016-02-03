library(caret)
library(AppliedPredictiveModeling)
library(C50)
library(ggbiplot)
library(doMC)
library(dplyr)
library(FUNctions)
library(Hmisc)

registerDoMC(4)
data(churn)

#################################################
#################################################
# have a quick look at data
# we see a very strong class imbalance, so we may want
# to take this into account for the metric to tune models to
# View(churnTrain)
describe(churnTrain)
summary(churnTrain)

# accuracy not the best metric- as I can get 85% accuracy by
# just guess no churn. More interested in sensitivity
# ie out of those we say churn, who actually churns?
table(churnTrain$churn)

# pull out predictive factors and the outcome
trainOutcome <- churnTrain$churn
testOutcome <- churnTest$churn
trainPred <- churnTrain[!names(churnTrain) == "churn"]
testPred <- churnTest[!names(churnTest) == "churn"]

#################################################
#################################################
# strategy: explore covariates, highlight any potential issues
# recode categorical as dummies
# calculate interactions between numerical predictors
# extract a full and reduced set of predictors
# build variety of models
# evaluate models with lift curves, calibration plots etc

##################################################
#################################################
# step 1: some visualisations

# firstly see what class the predictors are
vapply(churnTrain, class, character(1))

# seperate out factor, double (continuous) and integer (count) predictors
# train
facCols <- trainPred[, vapply(trainPred, is.factor, logical(1))]
numCols <- trainPred[, vapply(trainPred, is.double, logical(1))]
countCols <- trainPred[, vapply(trainPred, is.integer, logical(1))]

# test 
facColsT <- testPred[, vapply(testPred, is.factor, logical(1))]
numColsT <- testPred[, vapply(testPred, is.double, logical(1))]
countColsT <- testPred[, vapply(testPred, is.integer, logical(1))]

# CONTINUOUS NUMERIC PREDICTORS
# custom phil plots- if you dont have my FUNctions library you 
# cant make these. Also, you may want to build your own
# color theme- philTheme() probably isn't available to you!
plotListDouble <- ggplotListDens(numCols, trainOutcome)
ggMultiplot(plotListDouble, cols = 3)

# investigate correlated covariates- costs and minutes
ggplot(numCols, aes(x = total_day_minutes,
                    y = total_day_charge,
                    color = factor(trainOutcome))) +
  geom_point(alpha = 0.4, size = 4) +
  theme_bw() +
  scale_color_manual(values = philTheme()[c(4, 1)], name = "Churn") +
  theme(legend.position = c(0.1, 0.8),
        legend.text.align = 0) 

# investigate pairwise relation with caret::featurePlot
transparentTheme(trans = 0.1)
featurePlot(x = numCols,
            y = trainOutcome,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 2))

# COUNT DATA
plotListCount <- ggplotListHist(countCols, trainOutcome)
ggMultiplot(plotListCount, cols = 3)

# pairs
transparentTheme(trans = 0.1)
featurePlot(x = countCols,
            y = trainOutcome,
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 2))

# do we see any obvious class seperation for numeric?
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)

featurePlot(x = countCols,
            y = ifelse(trainOutcome== "yes", 1, 0),
            plot = "scatter",
            layout = c(4, 2))
featurePlot(x = numCols,
            y = ifelse(trainOutcome== "yes", 1, 0),
            plot = "scatter",
            layout = c(4, 2))

# FACTORS
plotListFac <- ggplotListBar(facCols, trainOutcome)
ggMultiplot(plotListFac, cols = 2)

# log odds
stateTab <- table(churnTrain$state, trainOutcome)
apply(stateTab, 1, function(x) log((x[1] / sum(x)) / (1 - (x[1] / sum(x)))))

# states are questionable value- could provide good info, but so few customers from each state.
# i would be inclined to conclude that need larger sample of data before reliably can use...

# tree based models really suited for degenerate distributions-
# we see in columns such as in number_vmail_message and total_day_minutes

##################################################
##################################################
# Lets act on what we discovered with visualistions.
# Remember everything done for train must be done for test!

# loose the charges, as correlated with minutes (totally uninformative
# to keep both)
# Tree models will split randomly with correlated predictors,
# so it makes models less interpretable.
# However, in terms of predictive power, they may
# not suffer if left in
numCols <- numCols[, !names(numCols) %in% c("total_day_charge",
                                            "total_eve_charge",
                                            "total_night_charge",
                                            "total_intl_charge")]
numColsT <- numColsT[, !names(numColsT) %in% c("total_day_charge",
                                              "total_eve_charge",
                                              "total_night_charge",
                                              "total_intl_charge")]

# not too worried about skewness- trees intrinsically
# resistant

#################################################
#################################################
# processing factors is the interesting bit for trees and rules.
# models can process as grouped or individual. We will have to make
# two versions of the input- one dummied, one containing factors.
# For the dummies, no need to remove uninformative (collinear) predictor
# that will arise, as models will simply ignore non informative predictors
# HOWEVER get some interpretability issues, as if have two correlated, will chose at random
# we will throw out one where only two levels, and for those with more than two keep all

# Note state will have 52 levels- this is ridiculous for categorical.
# Breiman's random forest will accept max 32 (for good reason)
# Therefore, for grouped, remove state as will cause issues in some models

##################################################
##################################################
# process categorical data

# dummy up
catDummies <- dummyVars(~. ,
                        data = facCols)

facTrans <- data.frame(predict(catDummies, facCols))
facTransT <- data.frame(predict(catDummies, facColsT))

# important to remember that for tree models:
# colinear will reduce interpretability
# but may not impact model peformance
# get rid of one for those binary factors, but where more
# than two levels leave them all in

# remove offending columns
facTrans <- facTrans[, !names(facTrans) %in% c("international_plan.no", 
                                               "voice_mail_plan.no")]
facTransT <- facTransT[, !names(facTransT) %in% c("international_plan.no", 
                                                  "voice_mail_plan.no")]

# rename
facTrans <- facTrans %>%
  dplyr::rename(voice_mail_plan = voice_mail_plan.yes,
                international_plan = international_plan.yes,
                area_code_510 = area_code.area_code_510,
                area_code_408 = area_code.area_code_408,
                area_code_415 = area_code.area_code_415)

facTransT <- facTransT %>%
  dplyr::rename(voice_mail_plan = voice_mail_plan.yes,
                international_plan = international_plan.yes,
                area_code_510 = area_code.area_code_510,
                area_code_408 = area_code.area_code_408,
                area_code_415 = area_code.area_code_415)

# tree based models should be fairly resistant
# to non-informative predictors, so we wont
# worry about filtering NZV. 
# OR should we? lets try a few independent with state filtered

#################################################
# next step - lets combine all of our numerical predictors
numInput <- cbind(numCols, countCols)
numInputT <- cbind(numColsT, countColsT)

# combine with categorical- independent categories
trainInputI <- cbind(numInput, facTrans)
testInputI <- cbind(numInputT, facTransT)

# independent categories with state removed
trainInputI2 <- trainInputI[, c(1:11, 63:67)]
testInputI2 <- testInputI[, c(1:11, 63:67)]

# grouped categories
trainInputG <- cbind(numInput, facCols)
testInputG <- cbind(numInputT, facColsT)

# remove state as too many levels in category (limit for some models)
trainInputG <- trainInputG[, !names(trainInputG) %in% "state"]
testInputG <- testInputG[, !names(testInputG) %in% "state"]

# and there we go- all ready!
# of course even tree based models are best with only informative
# predictors- but much more resistant, and peformance much less
# degraded when uninformative predictors included

#################################################
#################################################
# fit models
# always set seeds before training for reproduceability

# set up train control
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)

#################################################
# bagged CART trees

# independent- full
set.seed(476)
treebagTuneI <- train(x = trainInputI,
                     y = trainOutcome,
                     method = "treebag",
                     nbagg = 50,
                     metric = "Sens",
                     trControl = ctrl)
treebagTuneI
#save(treebagTuneI, file = "treebagTuneI.Rdata")
#load("treebagTuneI.Rdata")

# predictions for test set: can do for class and probabilities
treebagTuneIPred <- predict(treebagTuneI,
                            newdata = testInputI)
treebagTuneIPredProb <- predict(treebagTuneI,
                                newdata = testInputI, type = "prob")

# sensitivity of 0.710 on test set
confusionMatrix(data = treebagTuneIPred, reference = testOutcome)

# independent - reduced
set.seed(476)
treebagTuneI2 <- train(x = trainInputI2,
                      y = trainOutcome,
                      method = "treebag",
                      nbagg = 50,
                      metric = "Sens",
                      trControl = ctrl)
treebagTuneI2
#save(treebagTuneI2, file = "treebagTuneI2.Rdata")
#load("treebagTuneI2.Rdata")

# predictions for test set: can do for class and probabilities
treebagTuneI2Pred <- predict(treebagTuneI2,
                            newdata = testInputI2)
treebagTuneI2PredProb <- predict(treebagTuneI2,
                                newdata = testInputI, type = "prob")

# sens of 0.714 on test set
confusionMatrix(data = treebagTuneI2Pred, reference = testOutcome)

# grouped
set.seed(476)
treebagTuneG <- train(x = trainInputG,
                      y = trainOutcome,
                      method = "treebag",
                      nbagg = 50,
                      metric = "Sens",
                      trControl = ctrl)

#save(treebagTuneG, file = "treebagTuneG.Rdata")
#load("treebagTuneG.Rdata")

treebagTuneGPred <- predict(treebagTuneG,
                            newdata = testInputG)
treebagTuneGPredProb <- predict(treebagTuneG,
                                newdata = testInputG, type = "prob")

# sensitivity of 0.714 on test set
confusionMatrix(data = treebagTuneGPred, reference = testOutcome)

#################################################
# random forest

# independent - full
mtryValues <- c(4, 8, 13, 20, 30, 40)
set.seed(476)
rfTuneI <- train(x = trainInputI,
                y = trainOutcome,
                method = "rf",
                ntree = 1500,
                tuneGrid = data.frame(mtry = mtryValues),
                importance = TRUE,
                metric = "Sens",
                trControl = ctrl)

#save(rfTuneI, file = "rfTuneI.Rdata")
#load("rfTuneI.Rdata")

rfTuneIPred <- predict(rfTuneI,
                       newdata = testInputI)
rfTuneIPredProb <- predict(rfTuneI,
                           newdata = testInputI, type = "prob")

# sensitivity of 0.723 on test set
confusionMatrix(data = rfTuneIPred, reference = testOutcome)

# independent - reduced
mtryValues <- c(4, 8, 13)
rfTuneI2 <-  train(x = trainInputI2,
                   y = trainOutcome,
                   method = "rf",
                   ntree = 1500,
                   tuneGrid = data.frame(mtry = mtryValues),
                   importance = TRUE,
                   metric = "Sens",
                   trControl = ctrl)

#save(rfTuneI2, file = "rfTuneI2.Rdata")
#load("rfTuneI2.Rdata")

rfTuneI2Pred <- predict(rfTuneI2,
                       newdata = testInputI2)
rfTuneI2PredProb <- predict(rfTuneI2,
                           newdata = testInputI2, type = "prob")

# sensitivity of 0.732 on test set
confusionMatrix(data = rfTuneI2Pred, reference = testOutcome)

# grouped
mtryValues <- c(4, 6, 10, 13)
set.seed(476)
rfTuneG <- train(x = trainInputG,
                 y = trainOutcome,
                 method = "rf",
                 ntree = 1500,
                 tuneGrid = data.frame(mtry = mtryValues),
                 importance = TRUE,
                 metric = "Sens",
                 trControl = ctrl)

#save(rfTuneG, file = "rfTuneG.Rdata")
#load("rfTuneG.Rdata")

rfTuneGPred <- predict(rfTuneG,
                       newdata = testInputG)
rfTuneGPredProb <- predict(rfTuneG,
                           newdata = testInputG, type = "prob")

# sensitivity of 0.723 on test set
confusionMatrix(data = rfTuneGPred, reference = testOutcome)

#################################################
# boosted trees

gbmGrid <- expand.grid(interaction.depth = c(5, 7, 9),
                       n.trees = (seq(1, 13, 4))*100,
                       shrinkage = c(0.01, 0.03),
                       n.minobsinnode = 10)

# independent -full
set.seed(476)
gbmTuneI <- train(x = trainInputI,
                 y = trainOutcome,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 metric = "Sens",
                 verbose = FALSE,
                 trControl = ctrl)

gbmTuneI
#save(gbmTuneI, file = "gbmTuneI.Rdata")
#load("gbmTuneI.Rdata")

gbmTuneIPred <- predict(gbmTuneI,
                       newdata = testInputI)
gbmTuneIPredProb <- predict(gbmTuneI,
                           newdata = testInputI, type = "prob")

# sens = 0.732
confusionMatrix(data = gbmTuneIPred, reference = testOutcome)

# independent - reduced
set.seed(476)
gbmTuneI2 <- train(x = trainInputI2,
                  y = trainOutcome,
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  metric = "Sens",
                  verbose = FALSE,
                  trControl = ctrl)

gbmTuneI2
#save(gbmTuneI2, file = "gbmTuneI2.Rdata")
#load("gbmTuneI2.Rdata")

gbmTuneI2Pred <- predict(gbmTuneI2,
                        newdata = testInputI2)
gbmTuneI2PredProb <- predict(gbmTuneI2,
                            newdata = testInputI2, type = "prob")

# sens = 0.732
confusionMatrix(data = gbmTuneI2Pred, reference = testOutcome)

# grouped categories
set.seed(476)
gbmTuneG <- train(x = trainInputG,
                  y = trainOutcome,
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  metric = "Sens",
                  verbose = FALSE,
                  trControl = ctrl)

#save(gbmTuneG, file = "gbmTuneG.Rdata")
#load("gbmTuneG.Rdata")

gbmTuneGPred <- predict(gbmTuneG,
                        newdata = testInputG)
gbmTuneGPredProb <- predict(gbmTuneG,
                            newdata = testInputG, type = "prob")

# sens 0.732
confusionMatrix(data = gbmTuneGPred, reference = testOutcome)

# could consider other boosting- extereme gradient boosting (XGB)
# is currently very popular (but takes forever to run!)

#################################################
# C5.0 
c50Grid <- expand.grid(trials = c(1:9, (1:10)*10),
                       model = c("tree", "rules"),
                       winnow = c(TRUE, FALSE))

# independent - full
set.seed(476)
c50TuneI <- train(x = trainInputI,
                 y = trainOutcome,
                 method = "C5.0",
                 tuneGrid = c50Grid,
                 verbose = FALSE,
                 metric = "Sens",
                 trControl = ctrl)

#save(c50TuneI, file = "c50TuneI.Rdata")
#load("c50TuneI.Rdata")

c50TuneIPred <- predict(c50TuneI,
                        newdata = testInputI)
c50TuneIPredProb <- predict(c50TuneI,
                            newdata = testInputI, type = "prob")

# sensitvity of 0.737
confusionMatrix(data = c50TuneIPred, reference = testOutcome)

# independent - reduced
set.seed(476)
c50TuneI2 <- train(x = trainInputI2,
                  y = trainOutcome,
                  method = "C5.0",
                  tuneGrid = c50Grid,
                  verbose = FALSE,
                  metric = "Sens",
                  trControl = ctrl)

#save(c50TuneI2, file = "c50TuneI2.Rdata")
#load("c50TuneI2.Rdata")

c50TuneI2Pred <- predict(c50TuneI2,
                        newdata = testInputI2)
c50TuneI2PredProb <- predict(c50TuneI2,
                            newdata = testInputI2, type = "prob")

# sensitvity of 0.732
confusionMatrix(data = c50TuneI2Pred, reference = testOutcome)

# grouped
set.seed(476)
c50TuneG <- train(x = trainInputG,
                 y = trainOutcome,
                 method = "C5.0",
                 tuneGrid = c50Grid,
                 verbose = FALSE,
                 metric = "Sens",
                 trControl = ctrl)

#save(c50TuneG, file = "c50TuneG.Rdata")
#load("c50TuneG.Rdata")


c50TuneGPred <- predict(c50TuneG,
                        newdata = testInputG)
c50TuneGPredProb <- predict(c50TuneG,
                            newdata = testInputG, type = "prob")

# sensitvity of 0.683
confusionMatrix(data = c50TuneGPred, reference = testOutcome)

#################################################
#################################################
# gather results for comparison

models <- list(bagTreeI = treebagTuneI,
               bagTreeI2 = treebagTuneI2,
               bagTreeG = treebagTuneG,
               rfI = rfTuneI,
               rfI2 = rfTuneI2,
               rfG = rfTuneG,
               gbmI = gbmTuneI,
               gbmI2 = gbmTuneI2,
               gbmG = gbmTuneG,
               c50I = c50TuneI,
               c50I2 = c50TuneI2,
               c50G = c50TuneG)

# resamples are fairly correleated
resamp <- resamples(models)
modelCor(resamp, metric = "Sens")
bwplot(resamp)
splom(resamp, metric = "Sens")

# statistical test
t.test(resamp$values$`gbmI2~Sens`,
       resamp$values$`c50G~Sens`,
       paired = TRUE)

# top two models (by resamples) are gbmI2 and gbmG
# but lets see how different classes of model look
plot(varImp(gbmTuneI2))
plot(varImp(rfTuneI2))


# pull out results so can look at lift curves and calibration
# need predicted probabilities for this
results <- data.frame(bagTreeI = treebagTuneIPredProb$yes,
                      bagTreeI2 = treebagTuneI2PredProb$yes,
                      bagTreeG = treebagTuneGPredProb$yes,
                      rfI = rfTuneIPredProb$yes,
                      rfI2 = rfTuneI2PredProb$yes,
                      rfG = rfTuneGPredProb$yes,
                      gbmI = gbmTuneIPredProb$yes,
                      gbmI2 = gbmTuneI2PredProb$yes,
                      gbmG = gbmTuneGPredProb$yes,
                      c50I = c50TuneIPredProb$yes,
                      c50I2 = c50TuneI2PredProb$yes,
                      c50G = c50TuneGPredProb$yes,
                      class = testOutcome)


# calibration curves
# to make max's pretty plots use bookTheme() from APM package
trellis.par.set(bookTheme())
calCurve <- calibration(class ~ gbmG + gbmI + gbmI2,
                        data = results)
calCurve
# seem somewhat poorly calibrated,
# ie cannot really interpret probabilities as real probabilities
xyplot(calCurve,
       auto.key = list(columns = 3))

# lift curves
liftCurve <- lift(class ~ gbmG + gbmI + gbmI2,
                  data = results)
liftCurve
xyplot(liftCurve,
       auto.key = list(columns = 2,
                       lines = TRUE,
                       points = FALSE))

# need 12.7% to find 80% of churners
# lift plot plots CumTestedPct (x-axis) vs CumEventPct(y-axis)
