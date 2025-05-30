# Task:

# 1.Find the senator whose voting record is closest to Rhode Island senator, Lincoln Chafee.
# 2.Find the senator who disagrees most with Pennsylvania senator, Rick Santorum.
# 3.Vermont senator Jim Jeffords was an Independent. Choose 5 Democratic and 5 Republican (or more) senators. You should try to do this randomly. Compare Jefford's record with each of these 10 senators. Would you classify Jeffords as closer to the Democrats or a Republicans?


# import libraries
import pandas as pd
import numpy as np

#  reading and storing the data set
df = pd.read_csv("senate_votes.csv")

# define a function to calculate the cosine distance
def cosine_dis(a,b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0  or norm_b == 0:
        return 1
    cos = np.dot(a,b) / (norm_a * norm_b)
    dist = 1 - cos
    return dist

# Task1: storing the values of Lincoln Chafee
lincoln = df [(df['Name']=='Chafee')&(df['State']=='RI')]

#storing the cosine distance of Lincoln with others
ans=[cosine_dis(lincoln.iloc[:,3:49], df.iloc[i,3:49]) for i in range(len(df))]

##Executing this loop also does the same thing as above
# for i in range(len(df)):
#     row_i = lincoln.iloc[:,3:49]
#     row_j = df.iloc[i,3:49]
#     x=cosine_dis(row_i,row_j)
#     ans.append(x)

k=1
index=0

#checks the least cosine distance
for i in range(len(ans)):
    if df.iloc[i]['Name'] == 'Chafee' and df.iloc[i]['State']=='RI':#because cosine dist with himself will be 0
        continue
    elif ans[i]<k:
        index = i
        k=ans[i]

df.iloc[index, 0]

# Task2: storing the values of Rick Santorum
rick = df[(df['Name']=='Santorum')&(df['State']=='PA')]

#storing the cosine distances of Lincoln with others 
ans2=[cosine_dis(rick.iloc[:,3:49],df.iloc[i,3:49]) for i in range(len(df))]

##Executing this loop also does the same thing as above
# for i in range(len(df)):
#     row_i = rick.iloc[:,3:49]

#     row_j = df.iloc[i,3:49]
#     x=cosine_dis(row_i,row_j)
#     ans2.append(x)

p=0
index2=0

#checks the largest cosine distance
for i in range(len(ans2)):
    if df.iloc[i]['Name'] == 'Santorum' and df.iloc[i]['State'] == 'PA':
        continue
    elif ans2[i]>p:
        index2 = i
        p=ans2[i]

df.iloc[index2,0]


#Task 3: selecting five democrats
demo = df[df['Party']=='D'].sample(5, random_state=42)

#select five republicans
repub = df[df['Party']=='R'].sample(5, random_state=42)

#storing Jeffords
jeff = df[df['Name'] =='Jeffords']

#combining both the democrats and republicans
full = pd.concat([demo, repub])

ans3=[]
p=1
index3=0
classification_results =[]

#storing cosine distance of jeffords with randomly selected democrats and republicans
for i in range(len(full)):
    row_i = jeff.iloc[:, 3:49]
    row_j = jeff.iloc[i, 3:49]
    l= cosine_dis(row_i, row_j)
    ans3.append(l)
    classification_results.append({
        'Senator': full.iloc[i]['Name'],
        'State': full.iloc[i]['State'],
        'Party': full.iloc[i]['Party'],
        'Cosine Distance': l
        })
    
# storing the list according to cosine distances
classification_results = sorted(classification_results, key=lambda x: x['Cosine Distance'])

print("Comparison of Jim Jeffords' voting record with selected senators:")
for result in classification_results:
    print(f"Senator: {result['Senator']}, State: {result['State']}, Party: {result['Party']}, Cosine Distance: {result['Cosine Distance']}")

if classification_results[0]['Party'] == 'D':
    print(f"\nClosest is {classification_results[0]['Senator']} whos is of Democratic Party so we can say Jeffords aligns towards Democrats")
else:
    print(f"\nClosest is {classification_results[0]['Senator']} whos is of Republican Party so we can say Jeffords aligns towards Republicans")
    
# (f"Farthest is {classification_results[9]['Senator']} whos is of Party {classification_results[9]['Party']}")




ans3=[]
ans4=[]

#stores the cosine distances of Jeffords with Democrats and Republicans separately  
for i in range(len(demo)):
    row_i = jeff.iloc[:,3:49]
    row_j = demo.iloc[i,3:49]
    row_k = repub.iloc[i,3:49]
    l=cosine_dis(row_i,row_j)
    b=cosine_dis(row_i,row_k)
    ans3.append(l)
    ans4.append(b)


sumr=0
sumd=0
avgr=0     
avgd=0

#calculates the averages of cosine distances of Democrats and Republicans
for i in range(len(ans3)):
    sumr+= ans4[i]
    sumd+= ans3[i]
    avgd = sumd/len(ans3)
    avgr = sumr/len(ans3)

#compares who's avg cosine distacne is lower
if avgd< avgr:
    print("Jim Jeffords is more aligned towards Democrats")
else:
    print("Jim Jeffords is more aligned towards Republicans")

