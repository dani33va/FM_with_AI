import os
import pandas as pd 


save= input("Select your saveFile (1,2,3...): ")

sourceFolderPath = r'Database\files'+ '\\' + save + 'Export'

inputPath= sourceFolderPath+'\\HTML'
outputPath= sourceFolderPath+ '\\data' 

#potential comes from the name of the file, with the shape: 'minP for group'-'maxP for group'-'number of players with fixed potential, at the beggining'
def addPotential(data, file):
    potStr = file.replace(".html", "")
    min,max,fixed= map(float, potStr.split("-"))
    data['maxPotential']=max
    data['minPotential']=min
    data.loc[0:fixed-1,'minPotential']=max
    return data
    
    

# Collect all HTML files in the directory and turn them into an array of df
dfArray=[]
for root, dirs, files in os.walk(inputPath):
    for file in files:
        if file.endswith('.html'):
            filePath=os.path.join(inputPath, file)
            dfIterative=pd.read_html(filePath)[0]            
            dfArray.append(addPotential(dfIterative, file))

#concat them into only one df
df=pd.concat(dfArray, ignore_index=True)

#Some cells need adjusting. 
df= df.drop(['Potential', 'SHP', 'Style','Short-term Plans', 'General Happiness'], axis=1)

# '€' get displayed as â¬ due to encoding in the html and avoids a correct parsing. They also contain comas.
columnsWithPricing=['Wage', 'AP', 'Min Fee Rls to Foreign Clubs', 'Min Fee Rls Clubs In Cont Comp', 'Min Fee Rls', 'Wage Contrib.', 'Max WD','Min WD' ]
df = df.convert_dtypes()
for col in columnsWithPricing:
    df[col] = df[col].str.split('â').str[0]
    df[col] = df[col].str.replace(",", "")
    df[col]= pd.to_numeric(df[col], errors='ignore')

#Columns listed bellow contain leters to represent millionsand thousends. Release clauses also contain dashes
columnsWithLetters= ["Min Fee Rls Clubs In Cont Comp", 'Min Fee Rls Clubs In Cont Comp', 'Min Fee Rls', 'Min Fee Rls to Foreign Clubs', 'AP']
for col in columnsWithLetters:
    for i in range(len(df)):
        cell=str(df.at[i, col])
        
        if "K" in cell:
            cell=cell.replace("K","")
            cell=pd.to_numeric(cell, errors='ignore')
            cell=cell*1000

        elif "M" in cell:
            cell=cell.replace("M","")
            cell=pd.to_numeric(cell, errors='ignore')
            cell=cell*1000000

        else:
            cell=pd.to_numeric(cell, errors='ignore')
            
        df.at[i,col]=cell

    df[col]=pd.to_numeric(df[col], errors='coerce') #we need coerce and ignore before because release clauses have both, dashes and letters

#Many columns contain dashses when null, so they need correcting
columnsWithDashes=['Max WD','Min WD', 'Av Rat', 'Yth Apps', 'AT Apps']
for col in columnsWithDashes:
    df[col]=pd.to_numeric(df[col], errors='coerce')

#I prefer to also modify two other columns as I have 2 'cons', consistency and the cons found by the scout
df=df.rename(columns={'Cons':'Consist', 'Cons.1':'Cons'})

''' Some quick and dirty tests
#pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', None)
print(df.loc[26:36, ['Min Fee Rls to Foreign Clubs','Av Rat', 'Min WD', 'Max WD', 'Wage Contrib.', 'AP', 'Max WD','Min WD', 'Av Rat', 'Wage', 'HR', 'Yth Apps', 'AT Apps'  ]])
#print(df["Min Fee Rls Clubs In Cont Comp"].value_counts())
print (df.dtypes)
'''

outFilePath=os.path.join(outputPath, (save+"save.parquet"))
df.to_parquet(outFilePath,index=False)
