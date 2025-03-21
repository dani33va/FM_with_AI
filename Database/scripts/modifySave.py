#It will modify the joined save adding more columns, now it will just transform one into the other
import pandas as pd
import os


input=(input('Select samples: '))
path= os.path.normpath('Database\\files\\'  + input +  'Export\\data\\' + input + 'save.parquet')
output=os.path.normpath('Database\\files\\'+ input +'Export\\data\\'   + input + 'saveFinal.parquet')


df=pd.read_parquet(path)
df.to_parquet(output, index=False)