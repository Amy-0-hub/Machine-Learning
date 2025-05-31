import numpy as np
import pandas as pd
import plotly.graph_objects as go

np.random.seed(42)
df = pd.DataFrame({'x1':np.random.uniform(-1, 1, size=(10_000)),\
                    'x2': np.random.uniform(-1, 1, size=(10_000)),})
df['y1'] = np.sin(np.pi*df['x1'])
df['y2'] = np.sin(np.pi*df['x2'])
print(df.head())

#constant hypothesis for each pair of points
df['g_cons'] = df[['y1','y2']].mean(axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x = df['x1'], y = df['y1'], mode = 'markers', name = 'f(x)'))
fig.add_trace(go.Scatter(x = df.loc[1,['x1','x2']], y = df.loc[1,['y1','y2']],\
                         mode = 'markers', marker = dict(size = 10), name = 'Generated points for the hypothesis'))
fig.add_trace(go.Scatter(x = df.loc[1,['x1','x2']], y = df.loc[1,['g_cons','g_cons']],\
                         mode = 'lines', marker = dict(size = 10), name = 'g1(x)'))
fig.update_layout(title = 'Example of a Constant Hypothesis')
fig.show()

#linear hypothesis for each pair of points
def findEqLine(x1, x2, y1, y2):
    '''
    This function takes 2 scalers for x1, and x2, and 2 scalers for y1, and y2.
    It will find the equation of a line that passes through these 2 points using matrix inverse.
    This function outputs the intercept and slope of the line (i.e intercept = w[0], slope = w[1])
    '''
    X = np.array([[1, x1], [1, x2]])
    w = np.linalg.pinv(X.transpose()@X)@X.transpose()@np.array([y1,y2])
    return(w)

# Run the above function for all 10,000 points. This will give us 10,000 slopes and intercepts.
for i in range(df.shape[0]):
    df.loc[i,'g_line_b'] = findEqLine(df.loc[i,'x1'], df.loc[i,'x2'], df.loc[i,'y1'], df.loc[i,'y2'])[:][0]
    df.loc[i,'g_line_m'] = findEqLine(df.loc[i,'x1'], df.loc[i,'x2'], df.loc[i,'y1'], df.loc[i,'y2'])[:][1]
print(df.head())

fig = go.Figure()
fig.add_trace(go.Scatter(x = df['x1'], y = df['y1'], mode = 'markers', name = 'f(x)'))
fig.add_trace(go.Scatter(x = df.loc[1,['x1','x2']], y = df.loc[1,['y1','y2']],\
                         mode = 'lines+markers', marker = dict(size = 10), name = 'g1(x)'))
fig.add_trace(go.Scatter(x = df.loc[2,['x1','x2']], y = df.loc[2,['y1','y2']],\
                         mode = 'lines+markers', marker = dict(size = 10),name = 'g2(x)'))
fig.update_layout(title = 'Examples of Linear Hypothesis')
fig.show()

#finding the avg hypothesis for the both models
g_cons_bar = df['g_cons'].mean()
g_line_m_bar = df['g_line_m'].mean()
g_line_b_bar = df['g_line_b'].mean()
print('For constant model, avg g(x)=', np.round(g_cons_bar,3), '\nFor linear model, avg g(x)=', np.round(g_line_m_bar,3),'x+',np.round(g_line_b_bar,3))

fig = go.Figure()
fig.add_trace(go.Scatter(x = df['x1'], y = df['y1'], mode = 'markers', name = 'f(x)'))
fig.add_trace(go.Scatter(x = df['x1'], y = np.repeat(g_cons_bar,10_000), mode = 'markers',\
             name = 'Avg. Const. Hyp.<br>g(x)=-0.002'))
fig.add_trace(go.Scatter(x = df['x1'], y = g_line_b_bar + g_line_m_bar*df['x1'], mode = 'markers',\
             name = 'Avg. Linear Hyp.<br>g(x)=0.786x-0.001'))
fig.update_layout(title = 'Average Hypothesis')
fig.show()

# bias^2 for the constant model
## bias^2 at x = (g_bar(x) - f(x))^2
## For constant model g_bar is the same at all x's
bias_cons_atX = (df['y1']-g_cons_bar)**2
#To find bias^2 we need to find E[bias^2 at x]. => Take expected value horizontally
bias_cons = np.mean(bias_cons_atX)
print('Bias Sq for constnat model is:', np.round(bias_cons,3))
print('Bias for constnat model is:', np.round(np.sqrt(bias_cons),3))

# bias^2 for the linear model
# Unlike constant model we have to evalute g_bar at every x
df['g_line_bar_atX'] = g_line_b_bar + g_line_m_bar*df['x1']
#Alternatively can use np.matmul(np.array([g_line_b_bar,g_line_m_bar]),np.array([np.ones(10000), df['x']]))
bias_linear_atX = (df['y1']-df['g_line_bar_atX'])**2
bias_linear = np.mean(bias_linear_atX)
print('Bias sq for linear model is:', np.round(bias_linear,3))
print('Bias for linear model is:', np.round(np.sqrt(bias_linear),3))

#variance for the constant model
# variance at each x =  var(x) = E[(g(x)-g_bar(x))^2]
# Constant model g_const1(x1) = g1 in the first row ,
# g_const2(x1) = g1 in the second row (it's the same no matter what the x is)
var_version1 = np.mean((df['g_cons']-g_cons_bar)**2)
var_version2 = np.var(df['g_cons'])
print('Variance for constant model is:', np.round(var_version1,3), np.round(var_version2,3))
fig = go.Figure()
fig.add_trace(go.Scatter(x = df['x1'], y = df['y1'], mode = 'markers', name = 'f(x)'))
fig.add_trace(go.Scatter(x = df['x1'], y = np.repeat(df.loc[1,'g_cons'], 10_000),\
                         mode = 'markers', name = 'g1(x)',\
                        marker = dict(size = 2, color = 'green')))
fig.add_trace(go.Scatter(x = df['x1'], y = np.repeat(df.loc[3,'g_cons'], 10_000),\
                         mode = 'markers', name = 'g2(x)',\
                        marker = dict(size = 2, color = 'green')))
fig.add_trace(go.Scatter(x = df['x1'], y = np.repeat(g_cons_bar, 10_000),\
                         mode = 'markers', name = 'g_bar(x)'))
fig.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="orange")
fig.update_layout(title = 'Variance Calculations<br>At each point x, var(x) = E[(g(x)-g_bar(x))^2]')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x = df['x1'], y = df['y1'], mode = 'markers', name = 'f(x)'))

fig.add_trace(go.Scatter(x = df['x1'], y = np.repeat(g_cons_bar, 10_000),\
                         mode = 'markers', name = 'g_bar(x)'))
fig.add_hrect(y0=g_cons_bar+var_version1, y1 =g_cons_bar-var_version1,\
             line_width = 0, fillcolor = 'darkgray', opacity = 0.5)
fig.update_layout(title = 'Upper and lower bounds of std for the constant model')
fig.show()

#variance for the linear model
g_linear_x = pd.DataFrame(np.matmul(np.array(df[['g_line_b','g_line_m']]),np.array([np.ones(10000), df['x1'] ])))
var_line_version1 = np.mean(g_linear_x.var())
temp = g_linear_x.sub(df['g_line_bar_atX'], axis = 'columns')**2
varAt_x = temp.mean()
var_line_version2 = np.mean(varAt_x)
print('Variance for linear model is:', np.round(var_line_version1,3), np.round(var_line_version2,3))

fig = go.Figure()
fig.add_trace(go.Scatter(x = df['x1'], y = df['g_line_bar_atX']+np.sqrt(varAt_x), mode = 'markers',
                        marker = dict(color='darkgray'),name='$\overline{g(x)}+\sigma$'))
fig.add_trace(go.Scatter(x = df['x1'], y = df['g_line_bar_atX']-np.sqrt(varAt_x), mode = 'markers',
                        marker = dict(color='darkgray'), name='$\overline{g(x)}-\sigma$'))
fig.add_trace(go.Scatter(x = df['x1'], y = df['y1'], mode = 'markers', name = 'f(x)'))
fig.update_layout(title = 'Upper and lower bounds of std for the linear model')
fig.show()