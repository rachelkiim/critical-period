# The datasets will be downloaded to the 'dataset' folder in the project root
# Original datasets are from https://drive.google.com/drive/folders/1JEOqxrhU_IhkdcRohdbuEtFETUxfNmNT,
# which is given by repository https://github.com/kakaoenterprise/Learning-Debiased-Disentangled

import os
import gdown
import zipfile

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset'))

download_link = {
    'cmnist': "1QnmRgeuf60vJSM1JU-Rwa3rvKa58IHkx",
    'cifar10c': "1RqT9kXyqS6MasdtwdBYMe3lreSklknap"
}

download_path = {
    'cmnist': os.path.join(base_dir, "cmnist.zip"),
    'cifar10c': os.path.join(base_dir, "cifar10c.zip")
}

unzip_path = {
    'cmnist': base_dir,
    'cifar10c': base_dir
}

def download_data(data_name):
    # Check if zip file exists
    if os.path.exists(download_path[data_name]):
        print(f"{data_name} data already exists")
    else:
        # Ensure the directory for the zip file exists
        os.makedirs(os.path.dirname(download_path[data_name]), exist_ok=True)

        try:
            gdown.download(id=download_link[data_name],
                           output=download_path[data_name],
                           quiet=False)
            print(f"Downloaded {data_name} data")
        except Exception as e:
            print(f"Failed to download {data_name} data: {e}")
            return

    # Check if unzip folder exists
    if os.path.exists(os.path.join(unzip_path[data_name], data_name)):
        print(f"{data_name} data already extracted")
    else:
        # Ensure the directory for extraction exists
        os.makedirs(unzip_path[data_name], exist_ok=True)

        try:
            with zipfile.ZipFile(download_path[data_name], 'r') as zip_ref:
                zip_ref.extractall(unzip_path[data_name])
            print(f"Extracted {data_name} data")
        except Exception as e:
            print(f"Failed to extract {data_name} data: {e}")

if __name__ == "__main__":
    download_data('cmnist')
    download_data('cifar10c')
