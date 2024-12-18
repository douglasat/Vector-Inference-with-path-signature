import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

df = pd.DataFrame(data)

df.index = ['a', 'b', 'c']

print(df)

# Original Series
s = pd.Series([1, 2, 3], name="bla")

df['bla'] = s
print(df)