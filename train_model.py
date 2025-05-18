#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
from hand_sign_recognition import get_model


# In[2]:


def main():
    parser = argparse.ArgumentParser(description='Train hand sign recognition model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset directory containing A-Z folders with images')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining even if model already exists')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        return
    
    print(f"Starting training process with dataset at {args.dataset_path}")
    print(f"Force retrain: {args.force}")
    
    try:
        model, labels = get_model(args.dataset_path, force_train=args.force)
        print(f"Training completed successfully!")
        print(f"Model recognizes {len(labels)} different hand signs: {', '.join(labels)}")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()


# In[ ]:




