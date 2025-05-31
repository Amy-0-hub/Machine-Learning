import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency

#input dataset
college_data = pd.read_csv("collegeData.csv")
college_data.info(verbose = True) 

#Question1: medians of GPA
#Null hypotheses: There is no statistically significant difference in median GPA between graduating and non-graduating groups.
#Alternative hypotheses: There is a statistically significant difference in median GPA between graduating and non-graduating groups.


# Split data into graduated and non-graduated groups
d1 = college_data.dropna(subset=['gradFlag','GPA'])
graduated_gpa = d1[d1['gradFlag'] == 1]['GPA']
non_graduated_gpa = d1[d1['gradFlag'] == 0]['GPA']


stat, p_value1 = mannwhitneyu(graduated_gpa, non_graduated_gpa)
print(f"Using Mann-Whitney U test \nMedians of GPA : U={stat}, p={p_value1}")

#Compare the p-value to the significance level 
alpha = 0.05
if p_value1 < alpha:
    print(f"Reject the null hypothesis in question 1")
else:
    print(f"Fail to reject the null hpothesis in question 1")


#Question2: Means of age at start
#Null hypotheses: There is no statistically significant difference in means of age at start between graduating and non-graduating groups.
#Alternative hypotheses: There is a statistically significant difference in means of age at start between graduating and non-graduating groups.

# Split data into graduated and non-graduated groups
d2 =  college_data.dropna(subset=['gradFlag', 'AgeAtStart'])
graduated_age = d2[d2['gradFlag'] == 1]['AgeAtStart']
non_graduated_age = d2[d2['gradFlag'] == 0]['AgeAtStart']


stat, p_value2 = ttest_ind(graduated_age, non_graduated_age, equal_var=False)
print(f"Using Two-Sample T-test \nMeans of age at start: t={stat}, p={p_value2}")

#Compare the p-value to the significance level 
if p_value2 < alpha:
    print(f"Reject the null hypothesis in question 2")
else:
    print(f"Fail to reject the null hpothesis in question 2")


#Question3: Medians of transfer GPA
#Null hypotheses: There is no statistically significant difference in medians of transfer GPA between graduating and non-graduating groups.
#Alternative hypotheses: There is a statistically significant difference in medians of transfer GPA between graduating and non-graduating groups.

# Split data into graduated and non-graduated groups
d3 =  college_data.dropna(subset=['gradFlag', 'TransferGPA'])
graduated_trans = d3[d3['gradFlag'] == 1]['TransferGPA']
non_graduated_trans = d3[d3['gradFlag'] == 0]['TransferGPA']


stat, p_value3 = mannwhitneyu(graduated_trans, non_graduated_trans)
print(f"Using Mann-Whitney U test \nMedians of Transfer GPA: U={stat}, p={p_value3}")

#Compare the p-value to the significance level 
if p_value3 < alpha:
    print(f"Reject the null hypothesis in question 3")
else:
    print(f"Fail to reject the null hpothesis in question 3")


#Question4: Means of transfer credits 
#Null hypotheses: There is no statistically significant difference in means of transfer credits between graduating and non-graduating groups.
#Alternative hypotheses: There is a statistically significant difference in means of transfer credits between graduating and non-graduating groups.

# Split data into graduated and non-graduated groups
d4 =  college_data.dropna(subset=['gradFlag','TransferCredits'])
graduated_credits =d4[d4['gradFlag'] == 1]['TransferCredits']
non_graduated_credits = d4[d4['gradFlag'] == 0]['TransferCredits']


stat, p_value4 = ttest_ind(graduated_credits, non_graduated_credits, equal_var=False)
print(f"Using Two-Sample T-test \nMeans of age at start: t={stat}, p={p_value4}")

#Compare the p-value to the significance level 
if p_value4 < alpha:
    print(f"Reject the null hypothesis in question 4")
else:
    print(f"Fail to reject the null hpothesis in question 4")


#Question5: Gender and graduation
#Null hypotheses: Gender and graduation are independent.
#Alternative hypotheses: Gender and graduation are not independent.

d5 = college_data.dropna(subset=['SexCode', 'gradFlag'])
gender_grad=pd.crosstab(d5['SexCode'], d5['gradFlag'])
chi2, p_value5, i,j=chi2_contingency(gender_grad)
print(f"Using Chi-Squared Test \nGender and graduation: chi2={chi2}, p={p_value5}")

#Compare the p-value to the significance level 
if p_value5 < alpha:
    print(f"Reject the null hypothesis in question 5")
else:
    print(f"Fail to reject the null hpothesis in question 5")

#Question6: Marital Status and graduation
#Null hypotheses: Marital Status and graduation are independent.
#Alternative hypotheses: Marital Status and graduation are not independent.

d6 = college_data.dropna(subset=['MaritalCode', 'gradFlag'])
marital_grad=pd.crosstab(d6['MaritalCode'], d6['gradFlag'])
chi2, p_value6, i,j=chi2_contingency(marital_grad)
print(f"Using Chi-Squared Test \nMarital Status and graduation: chi2={chi2}, p={p_value6}")

#Compare the p-value to the significance level 
if p_value6 < alpha:
    print(f"Reject the null hypothesis in question 6")
else:
    print(f"Fail to reject the null hpothesis in question 6")



#Question7: Previous education and graduation
#Null hypotheses: Previous education and graduation are independent.
#Alternative hypotheses: Previous education and graduation are not independent.

d7 = college_data.dropna(subset=['PrevEdCode', 'gradFlag'])
prevEd_grad=pd.crosstab(d7['PrevEdCode'], d7['gradFlag'])
chi2, p_value7, i,j=chi2_contingency(prevEd_grad)
print(f"Using Chi-Squared Test \nMarital Status and graduation: chi2={chi2}, p={p_value7}")

#Compare the p-value to the significance level 
if p_value7 < alpha:
    print(f"Reject the null hypothesis in question 7")
else:
    print(f"Fail to reject the null hpothesis in question 7")