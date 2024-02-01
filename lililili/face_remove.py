import os
import re

def removeUser(userId, path):
    """
    Remove all images associated with a user ID.
    """
    for filename in os.listdir(path):
        if re.match(f"Users-{userId}-\d+\.(\w+)$", filename):
            file_path = os.path.join(path, filename)
            # Remove the file
            os.remove(file_path)
            print(f"Removed: {file_path}")

path = './images/'
removeUser(0, path)
