from typing import Union, Callable
from dataclasses import dataclass
from multiprocessing.dummy import Pool

import PIL
from PIL import ImageFile, Image
import pandas as pd
import numpy as np
import pathlib
import requests
from io import BytesIO
from transformers import CLIPTokenizer



ImageFile.LOAD_TRUNCATED_IMAGES = True

def stream_image(url:str, threshold_size:int=512, debug:bool=True) -> Image:
    r"""
    stream image from internet
    
    args:
        url (:obj:`str`):
            image url
        rescale_size (:obj:`list` or `tuple`):
            width and height target
        threshold_size (:obj:`int`, *optional*, defaults to 512):
            minimum resolution
        debug (:obj:`bool`, *optional*, defaults to `False`):
            toggle print debug

    return: PIL.Image or None
    """
    
    try:
        # get images from the internet
        s = requests.Session()
        s.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        response = s.get(url, timeout=1)

        # Open the image using the Pillow library
        try:
            image = Image.open(BytesIO(response.content))
            
            # Check if the image is large enough 
            # if tru then return PIL image object
            if image.size[0] >= threshold_size and image.size[1] >= threshold_size:
                return image
            else:
                if debug:
                    print(f"Image {url} is too small, skipping.")
                pass
            
        except Exception as e:
            if debug:
                print(f"Error opening image {url}: {e}")
            pass

    except Exception as e:
        if debug:
            print(f"Error retrieving image {url}: {e}")
        pass

def process_image(
    rescale_size:Union[list, tuple],
    image:Image = None,
    image_path:str = None, 
    upper_bound:int = 10,
    debug:bool = False
) -> Union[np.array, tuple]:
    r"""
    scale the image resolution to predetermined resolution and return
    it as numpy
    
    args:
        rescale_size (:obj:`list` or `tuple`):
            width and height target
        image (:obj:`PIL.Image` defaults to `None`):
            image to process if `none` then it will try to read from `image_path`
        image_path (:obj:`str` defaults to `None`):
            path to file 
        upper_bound (:obj:`int`, *optional*, defaults to 10):
            major axis bound (not important, just set it as high as possible)
        debug (:obj:`bool`, *optional*, defaults to `False`):
            will return tuple (np.array, PIL.Image)

    return: np.array or (np.array, PIL.Image)
    """
    if image == None:
        image = Image.open(image_path)

    # find the scaling factor for each axis
    x_scale = rescale_size[0] / image.size[0]
    y_scale = rescale_size[1] / image.size[1]
    scaling_factor = max(x_scale, y_scale)

    # rescale image with scaling factor    
    new_scale = [round(image.size[0]*scaling_factor), round(image.size[1]*scaling_factor)]
    sampling_algo = PIL.Image.LANCZOS
    image = image.resize(new_scale, resample=sampling_algo)
    
    # get smallest and largest res from image
    minor_axis_value = min(image.size)
    minor_axis = image.size.index(minor_axis_value)
    major_axis_value = max(image.size)
    major_axis = image.size.index(major_axis_value)
    
    # warning
    if max(image.size) < max(rescale_size):
        print(f"[WARN] image {image_path} is smaller than designated batch, zero pad will be added")
    
    if minor_axis == 0:
        # left and right same crop top and bottom
        top = (image.size[1] - rescale_size[1])//2
        bottom = (image.size[1] + rescale_size[1])//2

        # remainder add
        bottom_remainder = (top  + bottom)
        # left, top, right, bottom
        image = image.crop((0, top, image.size[0], bottom))
    else:
        # top and bottom same crop the left and right
        left = (image.size[0] - rescale_size[0])//2
        right = (image.size[0] + rescale_size[0])//2
        # left, top, right, bottom
        image = image.crop((left, 0, right, image.size[1]))

    # cheeky resize to catch missmatch 
    image = image.resize(rescale_size, resample=sampling_algo)
    # for some reason np flip width and height
    np_image = np.array(image)
    # normalize
    np_image = np_image/127.5 - 1
    # height width channel to channel height weight
    np_image = np.transpose(np_image, (2,0,1))
    # add batch axis
    # np_image = np.expand_dims(np_image, axis=0)

    if debug:
        return (np_image, image)
    else:
        return (np_image)

def tokenize_text(
    tokenizer:CLIPTokenizer,
    text_prompt:list,
    max_length:int,
    batch_slice:int = 1,
) -> dict:
    r"""
    wraps huggingface tokenizer function with some batching functionality
    convert long token for example (1,1002) to (1,10,102)
    start and end token are extracted and reappended for each batch 
    
    args:
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        text_prompt (:obj:`list`):
            batch text to be tokenized
        max_length (:obj:`int`):
            maximum token before clipping
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (max_length-2) must be divisible by this value 

    return: 
        dict:
            {"attention_mask": np.array, "input_ids": np.array}
    """

    # check
    assert (max_length-2)%batch_slice == 0, "(max_length-2) must be divisible by batch_slice"

    text_input = tokenizer(
        text=text_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    max_length = tokenizer.model_max_length
    if batch_slice > 1:

        # ###[stack input ids]### #
        value = text_input["input_ids"]
        # strip start and end token
        # [start, token1, token2, ..., end] to 
        # [token1, token2, ..., tokenN]
        content = value[:,1:-1].reshape(-1, batch_slice, max_length-2)
        # store start and end token and then reshape it to be concatenated
        start = np.full(
            shape=(content.shape[0], content.shape[1], 1), 
            fill_value=[value[:,0][0]]
        )
        stop = np.full(
            shape=(content.shape[0], content.shape[1], 1), 
            fill_value=[value[:,-1][0]]
        )
        # concat start and end token
        # from shape (batch, 75*3+2)
        # to shape (batch, 3, 77)
        new_value = np.concatenate([start,content, stop], axis=-1)
        text_input["input_ids"] = new_value  

        # ###[stack attention mask]### #
        mask = text_input["attention_mask"]
          # strip start and end mask
        # [start, mask1, mask2, ..., end] to 
        # [mask1, mask2, ..., maskN]
        content = mask[:,1:-1].reshape(-1, batch_slice, max_length-2)
        # store start and end mask and then reshape it to be concatenated
        start = np.full(
            shape=(content.shape[0], content.shape[1], 1), 
            fill_value=[mask[:,0][0]]
        )
        # concat start and end mask
        # from shape (batch, 75*3+2)
        # to shape (batch, 3, 77)
        new_value = np.concatenate([start, start, content], axis=-1)
        text_input["attention_mask"] = new_value 

    return text_input

def generate_batch(
    process_image_fn:Callable[[str, tuple], np.array],
    tokenize_text_fn:Callable[[str, str, int], dict],
    tokenizer:CLIPTokenizer,
    dataframe:pd.DataFrame, 
    folder_path:str,
    image_name_col:str,
    caption_col:str,
    caption_token_length:int,
    tokenizer_path:str,
    width_col:str,
    height_col:str, 
    batch_slice:int=1
) -> dict:
    """
    generate a single batch for training.
    use this function in a for loop while swapping the dataframe batch
    depends on process_image and tokenize_text function

    args:
        process_image_fn (:obj:`Callable`):
            process_image function
        process_image_fn (:obj:`Callable`):
            tokenize_text function
        tokenizer (:obj:`CLIPTokenizer`):
            tokenizer class
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        folder_path (:obj:`str`):
            path to image folder
        image_name_col (:obj:`str`):
            column name inside dataframe filled with image names
        caption_col (:obj:`str`):
            column name inside dataframe filled with text captions
        caption_token_length (:obj:`int`):
            maximum token before clipping
        tokenizer_path (:obj:`str`):
            path to file / hugging face path
        width_col (:obj:`str`):
            column name inside dataframe filled with bucket width of an image
        height_col (:obj:`str`):
            column name inside dataframe filled with bucket height of an image
        batch_slice (:obj:`int`, *optional*, defaults to 1):
            if greater than 1 it will slice the token into batch evenly
            (caption_token_length-2) must be divisible by this value 
    return: 
        dict:
            {
                "attention_mask": np.array, 
                "input_ids": np.array, 
                "pixel_values": np.array 
            }
    """
    # count batch size
    batch_size = len(dataframe)
    batch_image = []

    # ###[process image]### #
    # process batch sequentialy
    for x in range(batch_size):
        # get image name and size from datadrame
        image_name = dataframe.iloc[x][image_name_col]
        width_height = [dataframe.iloc[x][width_col], dataframe.iloc[x][height_col]]

        # grab iamge from path and then process it
        image_path = pathlib.Path(folder_path, image_name)
        image = process_image_fn(image_path=image_path, rescale_size=width_height)
        
        batch_image.append(image) 
    # stack image into neat array
    batch_image = np.stack(batch_image)
    # as contiguous array
    batch_image = np.ascontiguousarray(batch_image)

    # ###[process token]### #
    batch_prompt = dataframe.loc[:,caption_col].tolist()
    tokenizer_dict = tokenize_text_fn(
        tokenizer=tokenizer, 
        text_prompt=batch_prompt, 
        max_length=caption_token_length, 
        batch_slice=batch_slice
    )
    output = {}
    output["pixel_values"] = batch_image
    output["input_ids"] = tokenizer_dict.input_ids
    output["attention_mask"] = tokenizer_dict.attention_mask

    return output

@dataclass
class pil_image:
    image: Image
        
# TODO: docstring
def stream_data(
    df:pd.DataFrame,
    batch_size:int,
    url_col:str,
    width_col:str,
    height_col:str,
    text_col:str,
    hash_col:str,
    tokenizer:CLIPTokenizer,
    caption_token_length:int,
    batch_slice:int=1,
    min_res_axis:int = 512,
    debug:bool = True,
    check_col:int = "downloaded_at_batch",
    stream_image_fn:Callable[[str, int, bool], pil_image] = stream_image,
    process_image_fn:Callable[[str, tuple], np.array] = process_image,
    tokenize_text_fn:Callable[[str, str, int], dict] = tokenize_text,
    
) -> Union[dict, dict, pd.DataFrame]:
    
    # NOTE!
    # dataframe must come in pre shuffled!
    # dataframe must come in with the same resolution bucket!
    # could add some assertion check
    # width and height col must be a bucketing size
    
    # TODO:
    # add terminate signal to signal the training loop to stop training and save the model
    # threading download
    
    # assigning a new column that marks
    # if it's already downloaded or not
    # -1 is marked to download
    # -2 is failed
    # > 0 is batch number
    if not check_col in df.columns.tolist():
        df[check_col] = -1
        if debug:
            print("dataframe does not contain prior `check_col` creating one")
    elif debug:
        print("dataframe already has `check_col` in it")

    # TODO:
    # select rows that yet to be processed THIS SELECTION IS COSTLY!
    # refactor this to put it into separate bins instead
    # need to return this unselected df and only process the selected df
    # that will remove this boolean comparison overhead
    # imagine if i use this for the entire laion dataset lol
    # also stream process need to be refactored to so it returned 2 df instead of 1
    selected_df = df[df[check_col]==-1]
    unselected_df = df[df[check_col]!=-1]
    
    # keep track of the previous batch retrieval
    max_check_col_value = df[check_col].max()
       
    # batch size == numb of worker
    # strip the tail end so each worker have equal dataframe
    # only if it's not divisible by worker count
    if len(selected_df) % batch_size > 0:
        
        remainder_selected_df = selected_df.iloc[-(len(selected_df) % batch_size):] # modulo return 0 bug!
        selected_df = selected_df.iloc[:-(len(selected_df) % batch_size)]
        
    else:
        remainder_selected_df = pd.DataFrame()
    
    # store chunks here
    chunked_df = [] 
    chunk_length = len(selected_df) // batch_size    
    
    # this is the loop to break up the dataframe into chunks
    # divide the dataframe equally for each workers
    # [1,2,...,end] to
    # [1,2,...n][n+1,n+2,...m][m+1,m+2,...,end]   
    for x in range(batch_size):
        start = chunk_length * x
        end = start + chunk_length
        chunk = selected_df.iloc[start:end]
        chunked_df.append(chunk)
            
    # concurrent stream 
    def _stream_process(df:pd.DataFrame):
        # store streamed data here
        streamed_data = [] 
        
        # keep trying until the batch size is met
        counter = 0
        
        # separating image streaming loop from processing loop
        while len(streamed_data) == 0:

            selected_row = df.iloc[counter]
            # image url to retrieve from
            image_url = selected_row[url_col]
            
            # bucket width and height
            width_height = [
                selected_row[width_col],
                selected_row[height_col],
            ]
            
            # get image from the url and store it as 
            # a pil image object, preparing conversion to numpy
            image = stream_image_fn(
                url=image_url, 
                threshold_size=min_res_axis,
                debug=debug
            )
            
            # accosiated text from the image for the tokenizer
            text = selected_row[text_col],
            
            # get hash column (or any unique col) to be associated with image
            # usefull for storing the image while streaming
            hash_value = selected_row[hash_col],
            
            # only increment if succeed!
            if image != None:
                
                # store everything here
                image_text_pair = {
                    "image":image.convert('RGB'), # remove alpha channel 
                    "text":text, 
                    "hash":hash_value, 
                    "size":width_height
                }
                streamed_data.append(image_text_pair)
                
                # if succeed write the check col to keep track the order
                # for replicability purposes i hope
                df.iloc[counter, df.columns.get_loc(check_col)] = max_check_col_value + 1
            
            else: 
                # i think i should add error column to know 
                # why it couldn't get the image
                df.iloc[counter, df.columns.get_loc(check_col)] = -2
                
            # also this counter should be inside else statement
            counter = counter + 1

        # this is convoluted but my mind is hazy rn so let it be
        # returning both data dict and dataframe to be concatenated for the next step
        return (streamed_data[0], df) 
    
    
    with Pool(processes=len(chunked_df)) as pool:
        # Use the pool to process the tasks concurrently for the entire batch
        concurrent_stream = pool.map(_stream_process, chunked_df)
    
    # unpack output
    batch_image, chunked_df = zip(*concurrent_stream)
    
    ##processing batch##
    
    # store all numpy images & text to be concatenated here
    np_image_batch = []
    text_batch = []
    
    for streamed_data in batch_image:

        # grab iamge from path and then process it
        image = process_image_fn(image=streamed_data["image"], rescale_size=streamed_data["size"])
        text = streamed_data["text"]
        
        np_image_batch.append(image)
        text_batch.append(text[0])
        
    # stack image into neat array 
    np_image_batch = np.stack(np_image_batch)
    # placeboo but i think it feels faster lol
    np_image_batch = np.ascontiguousarray(np_image_batch)
    
    # tokenize text into integer representing the text
    tokenizer_dict = tokenize_text_fn(
        tokenizer=tokenizer, 
        text_prompt=text_batch, 
        max_length=caption_token_length, 
        batch_slice=batch_slice
    )
    
    output = {
        "pixel_values":np_image_batch,
        "input_ids":tokenizer_dict.input_ids,
        "attention_mask":tokenizer_dict.attention_mask
    }
            
    # recombine dataframe again for the next iteration
    reconcat_df = pd.concat( list(chunked_df) + [remainder_selected_df, unselected_df])
    
    return(batch_image, output, reconcat_df)
