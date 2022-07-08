# Titanic Machine Learning from Disaster

## Competition Link

https://www.kaggle.com/competitions/titanic/overview

## Objective

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, I used Machine Learning to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## Dataset description

The given data can be split into two categories, Categorical Fields and Continuous Fields

### Categorical fields: These fields have a certain fixed number of valid inputs. May be nominal, ordinal, ratio-based, or interval-based.

- survival: Survival (0 = No, 1 = Yes)
- pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- sex: Sex
- embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

### Continuous fields: These fields have any number of valid inputs within theoretical minimum and maximum values.

- Age: Age in years
- sibsp: # of siblings / spouses aboard the Titanic
- parch: # of parents / children aboard the Titanic
- fare: Passenger fare
