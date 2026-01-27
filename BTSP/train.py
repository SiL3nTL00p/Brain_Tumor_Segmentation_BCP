import torch 
from data_loader import BraTSMRIDataset
import torch.utils.data as DataLoader

# checks wether the gpu is avaialble or not
device = torch.device("" if torch.cuda.is_available() else "cpu")

# loads the dataset for the training 
dataset = BraTSMRIDataset("/Users/pmanthan/Desktop/BTSP_training_data")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

