import paramiko
from scp import SCPClient
import os
import shutil
import zipfile
from datetime import datetime
import uuid
from std_out import Print
import traceback
import dropbox
import os

def send_zipped_file(zip_file):
    app_key = "d72u41vwr0s1oo0"
    app_secret = "yx7u2fjyxuqx6sp"

    # Try to load saved token
    if os.path.exists("dropbox_token.txt"):
        with open("dropbox_token.txt", "r") as f:
            access_token = f.read().strip()
        dbx = dropbox.Dropbox(access_token)
    else:
        # First time - authenticate with browser
        auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(app_key, app_secret , token_access_type="offline")
        print("Go to:", auth_flow.start())
        auth_code = input("Enter code: ")
        oauth_result = auth_flow.finish(auth_code)
        access_token = oauth_result.access_token
        
        # Save token
        with open("dropbox_token.txt", "w") as f:
            f.write(access_token)
        
        dbx = dropbox.Dropbox(access_token)
    
    # Upload
    filename = os.path.basename(zip_file)
    with open(zip_file, "rb") as f:
        dbx.files_upload(f.read(), f"/{filename}")
    
    link = dbx.sharing_create_shared_link_with_settings(f"/{filename}")
    print(f"✅ Uploaded: {link.url}")
    return link.url


# send_zipped_file("run.bash")

# send_zipped_file(
#     key_file="C://users/Brayo/Downloads/my_c71.pem",
#     local_zip="names.rar"
# )



def collect_and_zip_files():
    """
    Collect all .txt, .csv, and .json files from the script's directory,
    copy them into a single new folder, and zip that folder.
    """
    # Script directory (absolute path)
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Create unique folder name
    unique_id = uuid.uuid4().hex[:6]  # short random ID
    folder_name = f"collected_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
    folder_path = os.path.join(cwd, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # File extensions we care about
    extensions = (".txt", ".csv", ".json")

    # Copy matching files into new folder (flat, no subfolders)
    for file in os.listdir(cwd):
        file_path = os.path.join(cwd, file)  # absolute path
        if file.endswith(extensions) and os.path.isfile(file_path):
            shutil.copy(file_path, folder_path)

    # Create zip file (flat structure based on the collected folder)
    zip_name = f"{folder_name}.zip"
    zip_path = os.path.join(cwd, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, file)  # use just filename inside zip

    Print(f"✅ Collected files are in: {folder_path}")
    Print(f"✅ Zipped archive created: {zip_path}")

    return folder_path, zip_path