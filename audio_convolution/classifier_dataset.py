from torch.utils.data import Dataset

# Define the LabeledMusicDataset class
class LabeledMusicDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        """
        Initialize the dataset with labeled data.
        
        Args:
            data (list): List of tuples containing tensors and their respective labels.
        """
        self.data = data
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Return a sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            tuple: Tuple containing the data tensor and its label.
        """
        return self.data[index][0], self.data[index][1]