# College Data Hypothesis Testing Project

This project analyzes real-world student data from a college dataset to investigate academic outcomes and behavioral trends. We use hypothesis testing to assess administrative assumptions about student performance, time-to-enroll behavior, and entrance exam results in relation to graduation status.


## Dataset

- **File**: `collegeData.csv`
- **Description**: Contains student demographic, academic performance, and enrollment data.

### Key Columns

| Category | Column Name | Description |
|----------|-------------|-------------|
| Demographics | `SexCode`, `MaritalCode`, `AgeAtStart` | Gender, marital status, and age at start |
| Academic Performance | `GPA`, `HoursAttempt`, `HoursEarned`, `gradFlag` | GPA, credit hours, graduation flag |
| Enrollment Details | `DaysEnrollToStart`, `TransferCredits`, `MinEFC` | Enrollment timing, transfer credits, expected family contribution |
| Exam | `TransferGPA`, `EntranceExamScore` | Transfer GPA and entrance scores |


## Research Questions

We test **six hypotheses** related to student outcomes using **one-sample t-tests** at the 5% significance level.


## Methodology

1. **Data Cleaning**:
   - Removed missing values for each variable tested.
   - Separated students into two groups: **graduates** and **non-graduates** 
2. **Statistical Testing**
3. **Reporting**:
   - Clearly stated **null** and **alternative** hypotheses.
   - Calculated test statistics and **p-values**.
   - Reported whether to **reject** or **fail to reject** the null.


## Built with
1. numpy
2. pandas
3. scipy.stats




## Results Summary

| Research Question | Group        | p-value | Decision                | Conclusion |
|------------------|--------------|---------|--------------------------|------------|
| Case1a             | Non-Grads    | 0.014   | Reject Null              | Supports claim |
| Case1b             | Grads        | 0.035   | Reject Null              | Supports claim |
| Case2a             | Non-Grads    | 0.497   | Fail to Reject Null      | Inconclusive |
| Case2b             | Grads        | 0.422   | Fail to Reject Nul       | Inconclusive |
| Case3a             | Non-Grads    | 0.002   | Reject Null              | Supports claim |
| Case3b             | Grads        | 0.335   | Fail to Reject Null      | Inconclusive |







