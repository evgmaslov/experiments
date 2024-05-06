from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from transformers.trainer_callback import TrainerCallback

class CheckpointManager():
    def __init__(self, temp_dir: str, hub_dir: str):
        self.temp_dir = temp_dir
        self.hub_dir = hub_dir

        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)

        if not "content" in temp_dir:
            self.should_save = True
            self.load_checkpoint()
        else:
            self.should_save = False
        
    def load_checkpoint(self):
        last_checkpoint_id = ""
        last_checkpoint_name = ""
        last_checkpoint_ind = 0
        file_list = self.drive.ListFile({'q': f"'{self.hub_dir}' in parents and trashed=false"}).GetList()
        for check in file_list:
            name = check["title"]
            id = check["id"]
            if not name.startswith("checkpoint-"):
                continue
            ind = check.replace("checkpoint-", "")
            ind = int(ind)
            if ind > last_checkpoint_ind:
                last_checkpoint_ind = ind
                last_checkpoint_name = name
                last_checkpoint_id = id
        
        new_check_path = os.path.join(self.temp_dir, last_checkpoint_name)
        os.makedirs(new_check_path)

        check_files = self.drive.ListFile({'q': f"'{last_checkpoint_id}' in parents and trashed=false"}).GetList()
        for f in check_files:
            name = f["title"]
            path = os.path.join(new_check_path, name)
            f.GetContentfile(path)
        
    def save_checkpoint(self):
        if not self.should_save:
            return
        last_checkpoint_name = ""
        last_checkpoint_ind = 0
        for check in os.listdir(self.temp_dir):
            if not name.startswith("checkpoint-"):
                continue
            name = check.replace("checkpoint-", "")
            ind = int(name)
            if ind > last_checkpoint_ind:
                last_checkpoint_ind = ind
                last_checkpoint_name = check
        self.clear_checkpoints(self.hub_dir)
        checkpoint_folder_id = self.create_gdrive_folder(last_checkpoint_name, self.hub_dir)
        for name in os.listdir(os.path.join(self.temp_dir, last_checkpoint_name)):
            file_meta = {
                "title":name,
                "parents":[checkpoint_folder_id],
            }
            local_path = os.path.join(self.temp_dir, last_checkpoint_name, name)
            file_drive = self.drive.CreateFile(file_meta)
            file_drive.SetContentFile(local_path)
            file_drive.Upload()

    def create_gdrive_folder(self, folder_name: str, parent_id: str) -> str:
        folder_meta = {
            "title":folder_name,
            "parents":[parent_id],
            'mimeType': 'application/vnd.google-apps.folder',
        }
        folder_drive = self.drive.CreateFile(folder_meta)
        folder_drive.Upload()
        file_list = self.drive.ListFile({'q': f"'{parent_id}' in parents and trashed=false"}).GetList()
        folder_id = ""
        for f in file_list:
            if f["title"] == folder_name:
                folder_id = f["id"]
                break
        return folder_id
    
    def clear_checkpoints(self, checkpoint_folder_id: str):
        file_list = self.drive.ListFile({'q': f"'{checkpoint_folder_id}' in parents and trashed=false"}).GetList()
        for f in file_list:
            if f["title"].startswith("checkpoint"):
                f.Delete()

class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    def on_save(self, args, state, control, eval_dataloader, **kwargs):
        self.checkpoint_manager.save_checkpoint()