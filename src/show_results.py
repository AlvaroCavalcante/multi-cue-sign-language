import matplotlib.pyplot as plt
import pandas as pd

df_history = pd.read_csv('src/history.csv')
print(df_history.head())

plt.plot(list(range(0, len(df_history))), df_history['accuracy'])
plt.show()
plt.plot(list(range(0, len(df_history))), df_history['loss'], color='orange')
plt.show()