import pandas as pd
import os

numberOfSaves=int(input("Enter the number of saves you want to make into a file: "))
root="Database\\files\\"
filePath=os.path.join(root,str(numberOfSaves) + "savesDatabase.parquet")
df=[]
for i in range(numberOfSaves):
    path=(root+ str(i+1) + "Export\\data\\" + str(i+1) + "save.parquet" )
    df.append(pd.read_parquet(path))

combined_df = pd.concat(df, ignore_index=True)


combined_df.to_parquet(filePath, index=False)

#combined_df.to_excel("1.xlsx", index=False)
