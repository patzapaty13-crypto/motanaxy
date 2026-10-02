import zipfile
import os

with zipfile.ZipFile('motanaxy_kaggle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('auto_evolve.py')
    
    for root, dirs, files in os.walk('motanaxy'):
        if '__pycache__' in root:
            continue
        for file in files:
            z.write(os.path.join(root, file))

print("Created motanaxy_kaggle.zip")
# List contents
with zipfile.ZipFile('motanaxy_kaggle.zip', 'r') as z:
    for name in z.namelist():
        print(f"  {name}")
