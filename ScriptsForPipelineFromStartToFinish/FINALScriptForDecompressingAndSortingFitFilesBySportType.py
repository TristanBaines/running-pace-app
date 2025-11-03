import os
import gzip 
import shutil
from fitparse import FitFile

input_dir = 'D:\\TaliaStravaData\\fit_files' # 'C:\\Users\\User\\Desktop\\Skripsie\\most_recent_export\\activities_fit_files'

output_base_dir = 'D:\\TaliaStravaData\\fit_files_sorted' # C:\\Users\\User\\Desktop\\Skripsie\\most_recent_export\\activities_sorted_fit'

os.makedirs(output_base_dir, exist_ok=True)

def decompress_fit_gz(source_path, dest_path):
    with gzip.open(source_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            f_out.write(f_in.read())

def get_sport_type(fit_path):
    fitfile = FitFile(fit_path)
    sport_type = None
    
    for record in fitfile.get_messages(): # iterate over messages to get the sport type
        for data in record:
            if data.name == 'sport':
                sport_type = data.value
                break
        if sport_type:
            break
    return sport_type


for filename in os.listdir(input_dir): # process all .fit.gz files in the input directory
    if filename.endswith('.fit.gz'):
        gz_path = os.path.join(input_dir, filename)
        fit_filename = filename[:-3]  # remove .gz extension
        fit_path = os.path.join(input_dir, fit_filename)

        
        decompress_fit_gz(gz_path, fit_path) # decompress the .fit.gz to .fit

         
        sport = get_sport_type(fit_path) # parse to get sport/activity type
        if sport is None:
            sport = 'unknown'

        print(f"File: {filename} => Sport: {sport}")

        sport_folder = os.path.join(output_base_dir, sport)
        os.makedirs(sport_folder, exist_ok=True)

        dest_fit_path = os.path.join(sport_folder, fit_filename)
        shutil.move(fit_path, dest_fit_path)

print("Processing complete.")