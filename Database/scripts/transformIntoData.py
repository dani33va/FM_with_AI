import os
from bs4 import BeautifulSoup
import pyarrow.parquet as pq
import pandas as pd 
import pyarrow

save= input("Select your saveFile (1,2,3...): ")
sourceFolderPath = r'Database\files'+ '\\' + save + 'Export'

inputPath= sourceFolderPath+'\\HTML'
outputPath= sourceFolderPath+ '\\data'
1
# Collect all HTML files in the directory
htmlList = []
for root, dirs, files in os.walk(inputPath):
    for file in files:
        if file.endswith('.html'):
            htmlList.append(file)


#function to extract the table from the file
def extract_table_data(file_name, index):
    file_path = os.path.join(inputPath, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')    
    headers = []
    #i just take the headers from the first file
    if (index==0):
        for th in table.find("tr").find_all("th"):
            headers.append(th.text.strip())
    data = []
    for row in rows[1:]:
        cells = row.find_all('td')
        row_data = [cell.text.strip() for cell in cells]
        data.append(row_data)
    
    return data, headers

# Function to export the csv
def export_array_to_parquet(data, file_name, save_directory):
    file_name += '.parquet'
    full_path = os.path.join(save_directory, file_name)
    
    # Ensure the directory exists, if not, create it
    os.makedirs(save_directory, exist_ok=True)
    
    # Convert the array (list of lists) to a pandas DataFrame
    # Assuming the first row contains column names (headers)
    df = pd.DataFrame(data[1:], columns=data[0])
    
    # Write the DataFrame to a Parquet file
    pq.write_table(pyarrow.Table.from_pandas(df), full_path)
    #df.to_excel("output.xlsx", index=False) #in case i want to check their values in an excel


def addPotential(data, file):
    potStr = file.replace(".html", "")
    min,max,fixed= (potStr.split("-"))
    for row in data:
        row.extend([min, max])
    for i in range(int(fixed)): #some players have a fixed valuation of their potential, they are at the beginning of the file, their min potential is their fixed one
        data[i][len(data[i])-2]=max
    
    return data

    
completeData=[]
for i in range(len(htmlList)):
    file=htmlList[i]
    print(file)
    data, headers = extract_table_data(file, i)
    data= addPotential(data,file)
    
    if (i==0):
        headers.extend(["minPotential","maxPotential"])
        #I have to modify one of the columns as i have 2 'cons', consistency and the cons found by the scout
        headers[13]="Consist"
        completeData.append(headers)

    completeData.extend(data)
    print(len(data))

print(len(completeData))
export_array_to_parquet(completeData, (save+"save"),outputPath)


    


