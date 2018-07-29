# Fraud Detection Case Study

The objective of this case study is to help a company identify fraudulent events by leveraging their data to build predictive models and by deploying a web app that can be used by their Fraud Detection team.

## Results
1. The fraud and non-fraud class distributions are significantly different...

![Classes Chart](https://github.com/mcarpanelli/Fraud-Prediction/blob/master/figures/classes.png?raw=true)

...which translates into very high model performance.

![ROC Chart](https://github.com/mcarpanelli/Fraud-Prediction/blob/master/figures/roc.png?raw=true)

2. The top features in terms of average contribution to information gains are:
    * Number of previous payouts
    * Time elapsed between the creation of the event and its payout
    * The proportion of capital letters in the title of an event
    * gts?
    * User age
    * Average ticket price

![Importances Chart](https://github.com/mcarpanelli/Fraud-Prediction/blob/master/figures/feature_importances.png?raw=true)

3. Features positively associated with the probability of a fraudulent event are:
    * The proportion of capital letters in the title of an event
    * gst
    * Event published on Facebook
    * Maximum ticket price

4. Features negatively associated with the probability of a fraudulent event are:
    * Number of previous payouts
    * Time elapsed between the creation of the event and its payout
    * Whether the venue name is blank
    * Average ticket price
    * User age
    * Number of channels where the event is published

![PDPs Chart](https://github.com/mcarpanelli/Fraud-Prediction/blob/master/figures/pdp.png?raw=true)

## Recommendations

* Payout history delivers the strongest signal for detecting fraudulent events: Keep collecting information on the past of your clients.
* Create real-time notifications to spot potential fraud in events that (i) forecast brief time elapsed between event creation and payout, or (ii) contain a high proportion of capital letters in their title.

## Process

### Timeline

* Day 1: Project scoping, Team direction, Model building
* Day 2: Web app building and deployment

### Data Science

* Definition of Target `Fraud`, based on which events had been labeled as `fraudster`, `fraudster_att` or `fraudster_event`.
* Data cleaning and feature engineering based on domain knowledge on fraudulent transactions.
    * Number of Previous Payouts (`num_previous_payouts`)
    * Time elapsed between the event creation and its payout (`event_created_to_payout`)
    * Proportion of capital letters in the event's title (`screaming_factor`)
* Balance classes using ADASYN (belongs to the SMOTE family), to correct model for the fact that fraudulent events represent only 9% of events.
* Classification Model pipeline creation and calibration.
    * Random Forest
    * Gradient Boosting
    * Adaboosting
* Model performance evaluation based on ROC, precision and accuracy. Special focus on precision since we are particularly interested in the positive predictive value for the company. The value of accurately predicting fraud is higher than the value of accurately predicting non-fraud. In other words, it costs more money for the company to miss fraudalent cases than to misclassify cases that are not fradulent.
* Set up Flask server, and configure infrastructure on AWS.

### Web Development
* Develop ReactJS application that:
    1. Renders events predicting the probability of fraud
    2. Provides user friendly interface to help the user quickly pickup and characterize fraud
