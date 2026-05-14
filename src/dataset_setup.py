import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Custom tqdm progress bar wrapper to work with urllib."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def setup_ptb_xl():
    # Define target directories and URLs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')

    # Official PhysioNet name for url and initial extraction
    official_name = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    zip_path = os.path.join(raw_data_dir, f"{official_name}.zip")
    zip_url = f"https://physionet.org/static/published-projects/ptb-xl/{official_name}.zip"

    # Clean Folder Name
    clean_folder_name = 'ptb-xl-dataset'
    final_target_folder = os.path.join(raw_data_dir, clean_folder_name)
    original_extracted_folder = os.path.join(raw_data_dir, official_name)
    # Ensure sure the data/raw directory exists
    os.makedirs(raw_data_dir, exist_ok=True)

    # Check if the unzipped folder already exists
    if os.path.exists(final_target_folder) and len(os.listdir(final_target_folder)) > 0:
        print(f"Dataset already exists and is unzipped at: {final_target_folder}")
        return

    # If not unzipped, check if we need to download the zip
    if not os.path.exists(zip_path):
        print("Zip file not found. Download 1.7 GB from PhysioNet...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(zip_url, filename=zip_path,reporthook=t.update_to)
        print("Download Complete!")
    else:
        print("Zip file found locally. Skipping download.")

    # Extract the zip file
    print("Extracting files... please wait.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, raw_data_dir)
    print("Extraction complete!")

    # Rename the extracted folder to clean name
    print(f"Renaming dataset folder to '{clean_folder_name}'...")
    if os.path.exists(original_extracted_folder):
        os.rename(original_extracted_folder, final_target_folder)

    # Clean up heavy zip file to save space
    print("Cleaning up the .zip file...")
    os.remove(zip_path)

    print("Dataset setup is fully complete and ready for preprocessing!")

if __name__ == "__main__":
    setup_ptb_xl()