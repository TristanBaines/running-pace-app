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
    # Iterate over messages to get the sport type
    for record in fitfile.get_messages():
        for data in record:
            if data.name == 'sport':
                sport_type = data.value
                break
        if sport_type:
            break
    return sport_type

# Process all .fit.gz files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.fit.gz'):
        gz_path = os.path.join(input_dir, filename)
        fit_filename = filename[:-3]  # Remove .gz extension
        fit_path = os.path.join(input_dir, fit_filename)

        # 1. Decompress the .fit.gz to .fit
        decompress_fit_gz(gz_path, fit_path)

        # 2. Parse to get sport/activity type
        sport = get_sport_type(fit_path)
        if sport is None:
            sport = 'unknown'

        print(f"File: {filename} => Sport: {sport}")

        # 3. Prepare output folder for this sport
        sport_folder = os.path.join(output_base_dir, sport)
        os.makedirs(sport_folder, exist_ok=True)

        # 4. Move or copy the decompressed .fit file
        dest_fit_path = os.path.join(sport_folder, fit_filename)
        shutil.move(fit_path, dest_fit_path)

        # Optional: If you want to also move the original .fit.gz file, uncomment:
        # dest_gz_path = os.path.join(sport_folder, filename)
        # shutil.move(gz_path, dest_gz_path)

print("Processing complete.")