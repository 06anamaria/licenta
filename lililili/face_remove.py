import os
import re

def removeUser(userId, path):
    """
    Remove all images associated with a given user ID from the specified directory.

    Parameters: 
        userId (int): The user ID of the user to be removed.
        path (str): Directory path containing face images.

    Returns: 
        None
    """
    # Iterate over all files in the directory
    for filename in os.listdir(path):
        # Check if the file belongs to the specified user
        if re.match(f"Users-{userId}-\d+\.(\w+)$", filename):
            # Construct the full file path
            file_path = os.path.join(path, filename)
            # Remove the file
            os.remove(file_path)
            print(f"Removed: {file_path}")

# Example usage
path = './images/'
removeUser(0, path)
