import pandas as pd
import statsmodels.api as sm

location = ''
data = pd.read_csv(location)
data = data.dropna(axis=1, how='all')
data = data.dropna(axis=0, how='any')
data = data.drop(axis=1, columns = ['audio_record_id', 'short_id', 'dataset', 'birthday', 'date left',
                                    'startOccurredOn', 'start left', 'type', 'Unnamed: 72', 'Unnamed: 73'])


X = data.drop(axis=1, columns = ['Age (m)']).astype(float)
X = (X-X.min()) / (X.max()-X.min())
X = sm.add_constant(X)
Y = data['Age (m)'].astype(float)

model = sm.OLS(Y, X).fit()
print(model.summary())

params = model.params
abs_sorted = params.iloc[params.abs().argsort()]
abs_sorted = abs(abs_sorted)
abs_sorted = 10 * (abs_sorted-abs_sorted.min()) / (abs_sorted.max()-abs_sorted.min())
