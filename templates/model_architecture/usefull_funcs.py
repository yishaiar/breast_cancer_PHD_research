import os
def folderExists(path,prompt = True):
  if not os.path.exists(path):
   # Create a new directory because it does not exist
   os.makedirs(path)
   if prompt:
    print("The new directory is created!")