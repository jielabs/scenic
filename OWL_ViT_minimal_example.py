#!/usr/bin/env python
# coding: utf-8

# # OWL-ViT minimal example
# 
# This Colab shows how to **load a pre-trained OWL-ViT checkpoint** and use it to
# **get object detection predictions** for an image.

# # Download and install OWL-ViT
# 
# OWL-ViT is implemented in [Scenic](https://github.com/google-research/scenic). The cell below installs the Scenic codebase from GitHub and imports it.

# In[1]:


# !rm -rf *
# !rm -rf .config
# !rm -rf .git
# !git clone https://github.com/google-research/scenic.git .
# !python -m pip install -q .
# !python -m pip install -r scenic/projects/baselines/clip/requirements.txt
# !echo "Done."


# In[2]:


import os

import jax
from matplotlib import pyplot as plt
import numpy as np
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b32, clip_l14
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from skimage import transform as skimage_transform


# # Choose config

# In[3]:


config = clip_b32.get_config()
# config = clip_l14.get_config()


# # Load the model and variables

# In[4]:


module = models.TextZeroShotDetectionModule(
    body_configs=config.model.body,
    normalize=config.model.normalize,
    box_bias=config.model.box_bias)


# In[5]:


config.init_from.checkpoint_path


# In[6]:


checkpoint_path = './clip_vit_b32_b0203fc'
# checkpoint_path = './clip_vit_l14_d83d374'

variables = module.load_variables(checkpoint_path)


# # Prepare image

# In[7]:


# filename = os.path.join(skimage.data_dir, 'astronaut.png')
filename = './images/dogpark.jpg'
filename = './images/trees.jpeg'
filename = './images/cows.jpeg'
filename = './images/straw.jpeg'
print(filename)


# In[8]:


# Load example image:
# filename = os.path.join(skimage.data_dir, 'astronaut.png')
image_uint8 = skimage_io.imread(filename)
image = image_uint8.astype(np.float32) / 255.0

# Pad to square with gray pixels on bottom and right:
h, w, _ = image.shape
size = max(h, w)
image_padded = np.pad(
    image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

# Resize to model input size:
input_image = skimage.transform.resize(
    image_padded,
    (config.dataset_configs.input_size, config.dataset_configs.input_size),
    anti_aliasing=True)


# In[9]:


print(input_image.shape)
plt.imshow(input_image)


# # Prepare text queries

# In[10]:


#text_query_line = input()
text_queries = ['human face', 'rocket', 'nasa badge', 'star-spangled banner']
text_queries = ['human', 'dog', 'tree']
text_queries = ['house', 'tree']
text_queries = ['cow']
text_queries = ['green strawberry', 'red strawberry', 'flower', 'tree']


# In[11]:


tokenized_queries = np.array([
    module.tokenize(q, config.dataset_configs.max_query_length)
    for q in text_queries
])

# Pad tokenized queries to avoid recompilation if number of queries changes:
tokenized_queries = np.pad(
    tokenized_queries,
    pad_width=((0, 100 - len(text_queries)), (0, 0)),
    constant_values=0)


# # Get predictions
# This will take a minute on the first execution due to model compilation. Subsequent executions will be faster.

# In[24]:


# Note: The model expects a batch dimension.
import time

start_ts = time.time()
predictions = module.apply(
    variables,
    input_image[None, ...],
    tokenized_queries[None, ...],
    train=False)

# Remove batch dimension and convert to numpy:
predictions = jax.tree_map(lambda x: np.array(x[0]), predictions )
print('inference time: %.2f' % (time.time() - start_ts))  # 5.31 for b32; 91.45 for l14


# In[16]:


predictions.keys()


# In[22]:


for key in predictions.keys():
    print(key, predictions[key].shape)
#    print(predictions[key])


# # Plot predictions

# In[20]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# https://i.imgur.com/1IWZX69.jpg

score_threshold = 0.02

logits = predictions['pred_logits'][..., :len(text_queries)]  # Remove padding.
scores = sigmoid(np.max(logits, axis=-1))
labels = np.argmax(predictions['pred_logits'], axis=-1)
boxes = predictions['pred_boxes']

fig, ax = plt.subplots(1, 1, figsize=(16, 16))
ax.imshow(input_image, extent=(0, 1, 1, 0))
ax.set_axis_off()

for score, box, label in zip(scores, boxes, labels):
  if score < score_threshold:
    continue
  cx, cy, w, h = box
  ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
          [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], 'r')
  ax.text(
      cx - w / 2,
      cy + h / 2 + 0.015,
      f'{text_queries[label]}: {score:1.2f}',
      ha='left',
      va='top',
      color='red',
      bbox={
          'facecolor': 'white',
          'edgecolor': 'red',
          'boxstyle': 'square,pad=.3'
      })


# In[ ]:




