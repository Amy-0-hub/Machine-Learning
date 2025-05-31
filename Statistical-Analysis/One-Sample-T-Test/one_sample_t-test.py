import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
#input dataset
college_data = pd.read_csv("collegeData.csv")
# Insepct the missing values
college_data.info()  
college_data.describe()

#Case 1: Do students who drop out tend to have lower transferred GPA compared to those who graduate?
#hypothesize (a) for students who do not graduate: The average transfer GPA is less than 2.75.
#hypothesize (b) for students who graduate: The average transfer GPA is greater than 2.8.

#get two groups: graduated students and non-graduated students
transfer =  college_data.dropna(subset=['TransferGPA', 'gradFlag'])
non_graduated = transfer[transfer['gradFlag'] == 0]
graduated = transfer[transfer['gradFlag'] == 1]


#check hypothesize (a) and (b)
t_stat1, p_value1 = ttest_1samp(non_graduated['TransferGPA'], 2.75, alternative='less')
print("NON-graduated average transfer GPA test:")
print(f" T-statistic: {t_stat1}\n P-value: {p_value1}")
alpha = 0.05
if p_value1 < alpha:
    print(f" They have enough evidence to reject the null hypothesis in case 1")
else:
    print(f"They have insufficient evidence to reject the null hpothesis in case 1")


t_stat2, p_value2 = ttest_1samp(graduated['TransferGPA'], 2.8, alternative='greater')
print("Graduated average transfer GPA test:")
print(f" T-statistic: {t_stat2}\n P-value: {p_value2}")
if p_value2 < alpha:
    print(f" They have enough evidence to reject the null hypothesis in case 1")
else:
    print(f"They have insufficient evidence to reject the null hpothesis in case 1")

#Case 2:Do students who drop out tend to have a shorter time gap between enrollment and the start of the semester compared to those who graduate? 
#hypothesize (a) for students who do not graduate: The average number of days between enrollment and the start of the semester is less than 71 days.
#hypothesize (b) for students who graduate: The average number of days between enrollment and the start of the semester is greater than 71 days.

#get two groups
enrollment = college_data.dropna(subset=['DaysEnrollToStart', 'gradFlag'])
non_graduated = enrollment[enrollment['gradFlag'] == 0]
graduated = enrollment[enrollment['gradFlag'] == 1]


#check hypothesize (a) and (b)
t_stat3, p_value3 = ttest_1samp(non_graduated['DaysEnrollToStart'], 71, alternative='less')
print("Average days from enrollment to semester start for graduates:")
print(f" T-statistic: {t_stat3}\n P-value: {p_value3}")
alpha = 0.05
if p_value3 < alpha:
    print(f" They have enough evidence to reject the null hypothesis in case 2")
else:
    print(f"They have insufficient evidence to reject the null hpothesis in case 2")


t_stat4, p_value4 = ttest_1samp(graduated['DaysEnrollToStart'], 71, alternative='greater')
print("Average days from enrollment to semester start for non-graduates:")
print(f" T-statistic: {t_stat4}\n P-value: {p_value4}")
if p_value4 < alpha:
    print(f" They have enough evidence to reject the null hypothesis in case 2")
else:
    print(f"They have insufficient evidence to reject the null hpothesis in case 2")

#Case 3:Do students who drop out tend to have lower entrance exam score compared to those who graduate?  
#hypothesize (a) for students who do not graduate: The average entrance exam score is less than 83.
#hypothesize (b) for students who graduate: The average entrance exam score is greater than 90.

#get two groups
entrance_exam = college_data.dropna(subset=['MaxENTEntranceScore', 'gradFlag'])
non_graduated = entrance_exam[entrance_exam['gradFlag'] == 0]
graduated = entrance_exam[entrance_exam['gradFlag'] == 1]


#check hypothesize (a) and (b)
average_score1 = non_graduated['MaxENTEntranceScore'].mean()
average_score2 = graduated['MaxENTEntranceScore'].mean()
print(f"Average entrance exam score of non-graduated students: {average_score1}")
print(f"Average entrance exam score of graduated students: {average_score2}")



t_stat5, p_value5 = ttest_1samp(non_graduated['MaxENTEntranceScore'], 83, alternative='less')
print("NON-graduated average entrance exam:")
print(f" T-statistic: {t_stat5}\n P-value: {p_value5}")
alpha = 0.05
if p_value5 < alpha:
    print(f" They have enough evidence to reject the null hypothesis in case 3")
else:
    print(f"They have insufficient evidence to reject the null hpothesis in case 3")


t_stat6, p_value6 = ttest_1samp(graduated['MaxENTEntranceScore'], 90, alternative='greater')
print("Graduated average entrance exam:")
print(f" T-statistic: {t_stat6}\n P-value: {p_value6}")
if p_value6 < alpha:
    print(f" They have enough evidence to reject the null hypothesis in case 3")
else:
    print(f"They have insufficient evidence to reject the null hpothesis in case 3")
