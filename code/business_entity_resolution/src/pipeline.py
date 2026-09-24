# src/pipeline.py
import pandas as pd
import numpy as np
import unicodedata
import re
import os

class EntityResolutionPipeline:
    def __init__(self, sample_size=50000, random_seed=42):
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # State variables to hold data across notebooks
        self.sample_s1 = None
        self.sample_s2 = None
        self.sample_s3 = None
        self.sample_gt = None

    @staticmethod
    def get_data_path(filename):
        """Bulletproof path resolver checking the current folder first."""
        possible_paths = [
            filename,  # Checks directly inside src/
            f"dataset/train/{filename}",
            f"../dataset/train/{filename}",
            f"../../dataset/train/{filename}"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Cannot find {filename}")

    @staticmethod
    def clean_text_fast(text):
        """Static utility for string normalization."""
        if pd.isna(text) or text == '': 
            return ""
        text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8').lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def load_and_sample(self):
        print("Resolving dataset paths...")
        
        # Now we just pass the file names directly
        s1_path = self.get_data_path("train_source1.tsv")
        s2_path = self.get_data_path("train_source2.tsv")
        s3_path = self.get_data_path("train_source3.tsv")
        gt_path = self.get_data_path("train_ground_truth.tsv")

        print("Loading datasets...")
        train_s1 = pd.read_csv(s1_path, sep="\t")
        train_s2 = pd.read_csv(s2_path, sep="\t")
        train_s3 = pd.read_csv(s3_path, sep="\t")
        gt = pd.read_csv(gt_path, sep="\t")

        print(f"Creating {self.sample_size}-entity sample...")
        np.random.seed(self.random_seed)
        sample_s1_ids = set(np.random.choice(gt['source1_entity_id'].values, size=self.sample_size, replace=False))
        self.sample_gt = gt[gt['source1_entity_id'].isin(sample_s1_ids)].copy()
        self.sample_s1 = train_s1[train_s1['entity_id'].isin(sample_s1_ids)].copy()

        matched_ids = set()
        for matches in self.sample_gt['matched_entity_ids'].dropna():
            for m in str(matches).split(','):
                if m.strip(): matched_ids.add(m.strip())

        self.sample_s2 = train_s2[train_s2['entity_id'].isin(matched_ids)].copy()
        self.sample_s3 = train_s3[train_s3['entity_id'].isin(matched_ids)].copy()

        print("Cleaning text strings...")
        for df in [self.sample_s1, self.sample_s2, self.sample_s3]:
            df['clean_name'] = df['business_name'].apply(self.clean_text_fast)
            df['clean_address'] = df['business_address'].apply(self.clean_text_fast)
            
        print("Data loaded, sampled, and cleaned successfully!")