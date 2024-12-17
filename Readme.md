# Image captioning using pre-trained Yolo v8m with LLAMA 3 - 8b

## Image Analysis : 
Started by getting objects information from 1000 image  from flicker captioned dataset


Example of information:
<!-- ```{
        {'objects_info': [{'name': 'bench_1',
    'bbox': [2484, 1984, 3652, 2550],
    'score': 0.8641608357429504,
    'color': 'midnightblue'}],
    'relationships': '',
    'setting': 'The image was taken during the night. It appears to be in a suburban area. ',
    'objects_count': {'bench': 1}} 
}; -->

Colors of objects where extracted by calculating **histogram** of each object image and get top 2  rgb values then search for the least distance from defined values withing webcolors package to get the name of the color 

## Fine tuning LLAMA 3  :
I used unsloth repo  to be able to use llama 3 with qunatization to make training faster  and finetuned using Lora technique

link of the repo : 

https://github.com/unslothai/unsloth


## Generating data :

Preprocessing.ipynb a notebook runed on Kaggle that utilize flicker dataset

 Generated 500 examples from flicker dataset  with what I could think of as useful information to the model.

getSpatialRelations.py contains the logic to apply on detected objects in a given image to get some context about relations like :
- A cat is on a left side of  a dog  
- A black dog is right to a white dog.



After finetuning I got these results 
![alt text](image.png)



Prompt for teaching LLAMA 3 -8B :
prompt :
```json
    {

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}

    }



## Examples of generated data :


        {
            #Example 1
        'instruction': 'Generate a detailed caption based on the given information about the image with hypothesizing any missing information.',
        'output': 'A blonde fisherman standing in the water prepares his hook .'},

        {'input': 'This image contains the following objects: bird_1 (confidence: 0.94, color: whitesmoke). The setting of the image is: The image maybe was taken during the night. It appears to be in a suburban area. ',

        'instruction': 'Generate a detailed caption based on the given information about the image with hypothesizing any missing information.',

        'output': 'A large white bird flies out of the water .'},

        #Example 2
        {'input': 'This image contains the following objects: dog_1 (confidence: 0.87, color: white), person_1 (confidence: 0.35, color: white). Relationships between objects: d o g_1   i s   t o   t h e   l e f t   o f   t h e   p e r s o n _ 1 .. The setting of the image is: The image maybe was taken during the daytime. It appears to be in a suburban area. ',
        'instruction': 'Generate a detailed caption based on the given information about the image with hypothesizing any missing information.',
        'output': 'A black dog plays with another animal .'}

```

## Handling edge cases : 

Using colorsExtractor.py we can extract most 3 colors in an image by using k-means algorithm and feed it to the LLM to generate a context even if no objects detected.

## Future work :

I was going to use **CLIP** model to generate embedding of image and calculate the distance between the generated candidates of captions from LLAMA model to choose best caption for the image , but had no time .


# Output on test images 
![Alt text](test_images/1.jpg)

prompt : 


caption : 


![Alt text](test_images/2.jpg)


prompt : 


caption : 


![Alt text](test_images/3.jpg)


prompt : 


caption : 

