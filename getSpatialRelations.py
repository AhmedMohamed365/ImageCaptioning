import torch
import numpy as np


# Function to compute the centroid of a bounding box
def get_centroid(bbox):
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

# Function to determine the spatial relationship between two bounding boxes
def get_relationship(obj1, obj2):
    x1, y1 = get_centroid(obj1['bbox'])
    x2, y2 = get_centroid(obj2['bbox'])
    
    # Check horizontal (left-right) and vertical (top-bottom) relationships
    if abs(x1 - x2) < 50 and abs(y1 - y2) < 50:
        return "near"
    elif x1 < x2:  # obj1 is to the left of obj2
        return "to the left of"
    elif x1 > x2:  # obj1 is to the right of obj2
        return "to the right of"
    elif y1 < y2:  # obj1 is above obj2
        return "above"
    elif y1 > y2:  # obj1 is below obj2
        return "below"
    
    return "near"  # Default case

# Gets relationship between object and each other object in a scene
def generate_caption(objects):
    caption_parts = []
    for index, obj1 in enumerate(objects):
        for obj2 in objects[index+1:]:
            relationship = get_relationship(obj1, obj2)
            # Form a sentence for the spatial relationship between objects
            caption_parts.append(f"The {obj1['name']} is {relationship} the {obj2['name']}.")

    return " ".join(caption_parts)

objects = [

    {"name": 'cat' , 'bbox' : [1,2,3,4]},
    
     {"name": 'dog' , 'bbox' : [-50,-100,3,4]}
]
# Generate and print caption
caption = generate_caption(objects)
print("Generated Caption:", caption)
