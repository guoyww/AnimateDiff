from torch.utils.data import Dataset, DataLoader
import os
import cv2

class TextVideoDataset(Dataset):
    def __init__(self, videos_dir, prompts_file, tokenizer, transform=None):
        self.videos_dir = videos_dir
        self.video_files = sorted(os.listdir(videos_dir))
        
        with open(prompts_file, 'r') as f:
            self.text_prompts = [line.strip() for line in f.readlines()]
        
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # Load video and convert to frames
        video_path = os.path.join(self.videos_dir, self.video_files[idx])
        video_frames = self._load_video(video_path)
        
        # Tokenize the text prompt
        prompt = self.text_prompts[idx]
        tokenized_prompt = self.tokenizer(prompt)
        
        if self.transform:
            video_frames = self.transform(video_frames)
        
        return video_frames, tokenized_prompt

    def _load_video(self, video_path, num_frames=16):
        # Load the video using OpenCV and extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Handle videos with fewer frames than num_frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return torch.stack([torch.tensor(f).float() for f in frames])

# Placeholder for the tokenizer
def dummy_tokenizer(text):
    # This is a placeholder. You would replace this with the actual tokenizer you have.
    return text

# Example usage
videos_dir = "./path_to_videos_directory"  # Replace with your directory path
prompts_file = "./path_to_prompts.txt"  # Replace with your prompts file path

dataset = TextVideoDataset(videos_dir, prompts_file, dummy_tokenizer)

# To check a single sample from the dataset
sample_video, sample_prompt = dataset[0]

sample_video, sample_prompt