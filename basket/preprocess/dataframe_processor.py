import pandas as pd
import numpy as np
import random

def discrete_scale_to_equal_area_old(
    dataframe:pd.DataFrame,
    image_height_col_name:str,
    image_width_col_name:str,
    new_image_height_col_name:str,
    new_image_width_col_name:str,
    maximum_area:int = 512 ** 2,
    nearest_multiple:int = 64,
    extreme_aspect_ratio_clip:float = 4.0,
    aspect_ratio_clamping:float = 2.0,
    return_with_helper_columns:bool = False
) -> pd.DataFrame:
    r"""
    scale the image resolution to nearest multiple value 
    with less or equal to the maximum area constraint

    note:
      this code assumes that the image is larger than maximum area
      if the image is smaller than maximum area it will get scaled up
    
    args:
      dataframe (:obj:`pd.DataFrame`):
        input dataframe
      image_height_col_name (:obj:`str`):
        target column height
      image_width_col_name (:obj:`str`):
        target column width
      new_image_height_col_name (:obj:`str`):
        column name for new height value
      new_image_width_col_name (:obj:`str`):
        column name for new width value
      maximum_area (:obj:`int`, *optional*, defaults to 512 ** 2):
        maximum pixel area to be compared with
      nearest_multiple (:obj:`int`, *optional*, defaults to 64):
        rounding value
      extreme_aspect_ratio_clip (:obj:`float`, *optional*, defaults to 4.0):
        drop images that have width/height or height/width 
        beyond threshold value
      aspect_ratio_clamping (:obj:`float`, *optional*, defaults to 2.0):
        crop images that have width/height or height/width 
        beyond threshold value 
      return_with_helper_columns (:obj:`bool`, *optional*, defaults to `False`):
        return pd.DataFramw with helper columns (for debugging purposes)

    return: pd.DataFrame
    """
    clamped_height = "_clamped_height"
    clamped_width = "_clamped_width"

    error_message = f"extreme_aspect_ratio_clip ({extreme_aspect_ratio_clip}) is less than aspect_ratio_clamping ({aspect_ratio_clamping})"
    assert extreme_aspect_ratio_clip > aspect_ratio_clamping , error_message

    # drop ridiculous aspect ratio
    dataframe = dataframe[dataframe[image_height_col_name] / dataframe[image_width_col_name] <=extreme_aspect_ratio_clip]
    dataframe = dataframe[dataframe[image_width_col_name] / dataframe[image_height_col_name] <=extreme_aspect_ratio_clip]

    # clamp aspect ratio
    dataframe[clamped_height] = dataframe[image_height_col_name]
    dataframe[clamped_width] = dataframe[image_width_col_name]
    loc_boolean_map = dataframe[clamped_height] / dataframe[clamped_width] >= aspect_ratio_clamping
    dataframe.loc[loc_boolean_map, clamped_height] = dataframe.loc[loc_boolean_map, clamped_width] * aspect_ratio_clamping
    loc_boolean_map = dataframe[clamped_width] / dataframe[clamped_height] >= aspect_ratio_clamping
    dataframe.loc[loc_boolean_map, clamped_width] = dataframe.loc[loc_boolean_map, clamped_height] * aspect_ratio_clamping

    #create square area scaling
    image_area = dataframe[clamped_height] * dataframe[clamped_width]
    image_area = (maximum_area / image_area) ** (1/2)

    # rescaling width and height
    new_height = (dataframe[clamped_height] * image_area) // nearest_multiple * nearest_multiple
    new_width = (dataframe[clamped_width] * image_area) // nearest_multiple * nearest_multiple

    # insert column to the dataframe
    dataframe[new_image_height_col_name] = new_height
    dataframe[new_image_width_col_name] = new_width

    # square special case
    loc_boolean_map = dataframe[clamped_height] == dataframe[clamped_width]
    dataframe.loc[loc_boolean_map, [new_image_width_col_name, new_image_height_col_name]] = maximum_area ** (1/2)
    
    # remove helper columns
    if not return_with_helper_columns:
      dataframe = dataframe.drop(columns=[clamped_height,clamped_width])

    return dataframe

def scale_by_minimum_axis(
    dataframe:pd.DataFrame,
    image_width_col:str,
    image_height_col:str,
    new_image_height_col:str,
    new_image_width_col:str,
    target_minimum_scale:int = 512,
    target_maximum_scale:int = 1024,
) -> pd.DataFrame:
    r"""
    scale the image resolution to nearest multiple value 
    with less or equal to the maximum area constraint

    note:
        this code assumes that the image is larger than maximum area
        if the image is smaller than maximum area it will get scaled up
    
    args:
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        image_width_col (:obj:`str`):
            target column width
        image_height_col (:obj:`str`):
            target column height
        new_image_width_col (:obj:`str`):
            column name for new width value
        new_image_height_col (:obj:`str`):
            column name for new height value
        target_minimum_scale (:obj:`int`, *optional*, defaults to 512):
            minimum axis pixel count (must be divisible by 64)
        target_maximum_scale (:obj:`int`, *optional*, defaults to 1024):
            maximum axis pixel count (must be divisible by 64)
        
    return: pd.DataFrame
    """

    min_axis = dataframe[[image_height_col, image_width_col]].min(axis=1)
    
    scale_factor =  target_minimum_scale / min_axis

    new_width = (round(dataframe[image_width_col] * scale_factor) // 64 * 64).astype(int)
    new_height = (round(dataframe[image_height_col] * scale_factor) // 64 * 64).astype(int)

    dataframe[new_image_height_col] = new_height
    dataframe[new_image_width_col] = new_width

    new_res_col = [new_image_width_col, new_image_height_col]

    dataframe[new_res_col] = dataframe[new_res_col].apply(
        np.clip, 
        a_min=target_minimum_scale, 
        a_max=target_maximum_scale
    )
    return dataframe

def discrete_scale_to_equal_area(
    dataframe:pd.DataFrame,
    image_width_col_name:str,
    image_height_col_name:str,
    new_image_width_col_name:str,
    new_image_height_col_name:str,
    max_res_area:int = 512 ** 2,
    bucket_lower_bound_res:int = 256,
    extreme_aspect_ratio_clip:float = 4.0,
    aspect_ratio_clamping:float = 2.0,
    return_with_helper_columns:bool = False
) -> pd.DataFrame:
    r"""
    scale the image resolution to nearest multiple value 
    with less or equal to the maximum area constraint

    note:
        this code assumes that the image is larger than maximum area
        if the image is smaller than maximum area it will get scaled up
    
    args:
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        image_width_col_name (:obj:`str`):
            target column width
        image_height_col_name (:obj:`str`):
            target column height
        new_image_width_col_name (:obj:`str`):
            column name for new width value
        new_image_height_col_name (:obj:`str`):
            column name for new height value
        max_res_area (:obj:`int`, *optional*, defaults to 512 ** 2):
            maximum pixel area to be compared with (must be a product of 
            w and h where w and h is divisible by 64)
        bucket_lower_bound_res (:obj:`int`, *optional*, defaults to 256):
            lowest possible pixel width/height for the image
        extreme_aspect_ratio_clip (:obj:`float`, *optional*, defaults to 4.0):
            drop images that have width/height or height/width 
            beyond threshold value
        aspect_ratio_clamping (:obj:`float`, *optional*, defaults to 2.0):
            crop images that have width/height or height/width 
            beyond threshold value 
        return_with_helper_columns (:obj:`bool`, *optional*, defaults to `False`):
            return pd.DataFramw with helper columns (for debugging purposes)

    return: pd.DataFrame
    """

    # local pandas column
    aspect_ratio_col_name = "_aspect_ratio"
    bucket_col_name = "_bucket_group"
    clamped_height = "_clamped_height"
    clamped_width = "_clamped_width"

    # ========[bucket generator section]======== #
    root_max_res = max_res_area ** (1/2)
    centroid = int(root_max_res)

    # a sequence of number that divisible by 64 with constraint
    w = np.arange(bucket_lower_bound_res // 64 * 64, centroid // 64 * 64 + 64, 64)
    # y=1/x formula with rounding down to the nearest multiple of 64 
    # will maximize the clamped resolution to maximum res area
    h = ((max_res_area/w) // 64 * 64).astype(int)
    # ========[/bucket generator section]======== #

    # drop ridiculous aspect ratio
    dataframe = dataframe[dataframe[image_height_col_name]/dataframe[image_width_col_name] <= extreme_aspect_ratio_clip]
    dataframe = dataframe[dataframe[image_width_col_name]/dataframe[image_height_col_name] <= extreme_aspect_ratio_clip]

    # ## portrait ## #
    # get portrait resolution
    # h/w
    width = dict(zip(list(range(len(w))), w))
    height = dict(zip(list(range(len(h))), h))
    # get portrait image only (height > width)
    portrait_image = dataframe.loc[dataframe[image_height_col_name]/dataframe[image_width_col_name]>=1].copy()
    # generate aspect ratio column (width/height)
    portrait_image[aspect_ratio_col_name] = portrait_image[image_height_col_name]/portrait_image[image_width_col_name]
    # group to the nearest mimimum portrait bucket aspect ratio and create a category column 
    portrait_image[bucket_col_name]=portrait_image[aspect_ratio_col_name].apply(lambda x: np.argmin(np.abs(x-(h/w))))
    # generate new column for new scaled portrait resolution
    portrait_image[new_image_height_col_name] = portrait_image[bucket_col_name].map(height).astype(int)
    portrait_image[new_image_width_col_name] = portrait_image[bucket_col_name].map(width).astype(int)

    # ## landscape ## #
    # get lanscape resolution
    # w_flip/h_flip
    h_flip = np.flip(w)
    w_flip = np.flip(h)
    width_flip = dict(zip(list(range(len(w), len(w_flip) + len(w))), w_flip))
    height_flip = dict(zip(list(range(len(h), len(h_flip) + len(h))), h_flip))
    
    # get landscape image only (width > height)
    landscape_image = dataframe.loc[dataframe[image_width_col_name]/dataframe[image_height_col_name]>1].copy()
    # generate aspect ratio column (width/height)
    landscape_image[aspect_ratio_col_name] = landscape_image[image_width_col_name]/landscape_image[image_height_col_name]
    # group to the nearest landscape bucket aspect ratio and create a category column 
    landscape_image[bucket_col_name]=landscape_image[aspect_ratio_col_name].apply(lambda x: np.argmin(np.abs(x-(w_flip/h_flip)))+len(w))
    # generate new column for new scaled landcape resolution
    landscape_image[new_image_width_col_name] = landscape_image[bucket_col_name].map(width_flip).astype(int)
    landscape_image[new_image_height_col_name] = landscape_image[bucket_col_name].map(height_flip).astype(int)

    

    dataframe = pd.concat([landscape_image, portrait_image])
    dataframe = dataframe.sort_index()

    # catch ungrouped and remove it
    dataframe = dataframe.dropna(axis=1)


    # drop local pandas column
    if not return_with_helper_columns:
        dataframe = dataframe.drop(columns=[
            aspect_ratio_col_name,
            bucket_col_name,
            ]
        )

    return dataframe

def resolution_bucketing_batch(
    dataframe:pd.DataFrame,
    image_height_col_name:str,
    image_width_col_name:str,
    seed:int = 0,
    bucket_batch_size:int = 8,
    bucket_group_col_name = "bucket_group"
) -> pd.DataFrame:
    r"""
    create aspect ratio bucket and batch it

    note:
        non full batch will get dropped

    args:
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        image_height_col_name (:obj:`str`):
            target column height
        image_width_col_name (:obj:`str`):
            target column width
        seed (:obj:`int`, *optional*, defaults to 0):
            rng seed for reproductibility
        bucket_batch_size (:obj: `int`, *optional*, default to 8):
            size of the bucket batch, non full batch will get dropped
        bucket_group_col_name (:obj:`str`):
            bucket column name to store randomized order

    return: pd.DataFrame
    """
    dataframe = dataframe.groupby([image_height_col_name, image_width_col_name])

    # store first batch for JAX to compile 
    first_batch = pd.DataFrame()
    remainder_batch = pd.DataFrame()

    # helper counter
    group_count = 0

    # increment bucket grouping order for the next group
    group_max_index = 0

    for group, data in dataframe:

        # helper counter
        group_count = group_count + 1

        # shuffle rows within group
        data = data.sample(frac=1, replace=False, random_state=seed)
        
        # create ordered index for generating bucket batch
        data = data.reset_index()
        data[bucket_group_col_name] = data.index // bucket_batch_size + group_max_index 

        # strip tail end bucket because it's not full bucket
        tail_end_length = len(data.loc[data[bucket_group_col_name] == data[bucket_group_col_name].max()])
        if tail_end_length < bucket_batch_size:
            data = data.iloc[:-tail_end_length,:]

        # build first batch for JAX to compile 
        first_batch = pd.concat([first_batch, data.iloc[-bucket_batch_size:,:]])

        # remainder batch
        data = data.iloc[:-bucket_batch_size,:]
        remainder_batch = pd.concat([remainder_batch, data])

        # increment bucket grouping order for the next group
        group_max_index = data[bucket_group_col_name].max() + 1
        
    # shuffling bucket
    bucket_order = remainder_batch[bucket_group_col_name]
    np.random.seed(seed)
    bucket_order_array = bucket_order.unique()
    np.random.shuffle(bucket_order_array)

    # replacing order of the bucket
    replace_dict = dict(zip(bucket_order.unique(), bucket_order_array)) 
    bucket_order = bucket_order.map(replace_dict)

    # shifting bucket index to make room for the first batch
    remainder_batch[bucket_group_col_name] = bucket_order + group_count

    # shuffling first batch bucket
    first_batch_order = first_batch[bucket_group_col_name]
    np.random.seed(seed)
    first_batch_order_array = first_batch_order.unique()
    np.random.shuffle(first_batch_order_array)

    # replacing order of the first batch bucket
    replace_dict = dict(zip(first_batch_order_array, list(range(len(first_batch_order_array))))) 
    first_batch[bucket_group_col_name] = first_batch_order.map(replace_dict)

    #combine both batch 
    dataframe = pd.concat([first_batch, remainder_batch])

    # restore original index back
    dataframe = dataframe.set_index(dataframe["index"], drop=True)

    # create multi level index for bucket s it can be accessed with loc
    dataframe = dataframe.set_index(dataframe[bucket_group_col_name], append=True)
    dataframe = dataframe.swaplevel(0,1)

    return dataframe

def resolution_bucketing_batch_with_chunking(
    dataframe:pd.DataFrame,
    image_height_col_name:str,
    image_width_col_name:str,
    seed:int = 0,
    bucket_batch_size:int = 8,
    repeat_batch:int = 20,
    bucket_group_col_name = "bucket_reso"
) -> pd.DataFrame:
    r"""
    create aspect ratio bucket and batch it but with additional chunk
    so swap overhead of jax compiled function is minimized

    note:
        non full batch will get dropped

    args:
        dataframe (:obj:`pd.DataFrame`):
            input dataframe
        image_height_col_name (:obj:`str`):
            target column height
        image_width_col_name (:obj:`str`):
            target column width
        seed (:obj:`int`, *optional*, defaults to 0):
            rng seed for reproducibility
        bucket_batch_size (:obj: `int`, *optional*, default to 8):
            size of the bucket batch, non full batch will get dropped
        repeat_batch (:obj: `int`, *optional*, default to 20):
            how many times batch with the same resolution is repeated
        bucket_group_col_name (:obj:`str`):
            bucket column name to store randomized order

    return: pd.DataFrame
    """

    # randomize the dataframe
    dataframe = dataframe.sample(frac=1, replace=False, random_state=seed)
    
    # create group from resolution 
    bucket_group = dataframe.groupby([image_width_col_name, image_height_col_name])
    
    first_batch = []
    remainder_batch = []
    tail_batch = []
    batch_counter = 0
    new_dataframe = pd.DataFrame()
          
    for bucket, data in bucket_group:
        
        # generate first batch
        first_sample = data.sample(bucket_batch_size, replace=False, random_state=seed)
        first_batch.append(first_sample)

        # remaining batch
        data = data[~data.index.isin(first_sample.index)]
        
        # strip tail end bucket because it's not full bucket
        tail_end_length = len(data) % bucket_batch_size
        if tail_end_length != 0:
            data = data.iloc[:-tail_end_length,:]
            
        # generate remainder and tail batch
        # this ensure resolution get repeated so jax does not have
        # to swap compiled cache back and forth too frequently
        mini_group = len(data) % (bucket_batch_size * repeat_batch)
        remainder_data = data
        if mini_group != 0:
            remainder_data = data.iloc[:-mini_group,:]
            
            # store the last bit 
            tail_data = data[~data.index.isin(remainder_data.index)]
            tail_batch.append(tail_data)
            
        # store mini group chunk
        for i in range(0, len(remainder_data), (bucket_batch_size * repeat_batch)):
            chunk = remainder_data.iloc[i:i + (bucket_batch_size * repeat_batch)]
            remainder_batch.append(chunk)
        
        # shuffle the list
        random.Random(seed+len(first_batch)).shuffle(first_batch)
        random.Random(seed+len(remainder_batch)).shuffle(remainder_batch)
        random.Random(seed+len(tail_batch)).shuffle(tail_batch)

    new_dataframe = pd.concat(first_batch + remainder_batch + tail_batch, ignore_index=True)
    
    return new_dataframe

def tag_suffler_to_comma_separated(tags:str, seed:int) -> str:
    r"""
    suffle and reformat tag from `this_is a_tag to_suffle`
    to `to suffle, a tag, this is`

    args:
        tags (:obj:`str`):
            tag string
        seed (:obj:`int`, *optional*, defaults to 0):
            rng seed for reproductibility

    return: str
    """

    tags = tags.replace(" ",",").replace("_"," ").split(",")
    random.Random(len(tags)+seed).shuffle(tags)
    tags = ", ".join(tags)
    return(tags)
  
def tag_suffler_to_space_separated(tags:str, seed:int) -> str:
    r"""
    suffle and reformat tag from `this_is a_tag to_suffle`
    to `to_suffle a_tag this_is`

    args:
        tags (:obj:`str`):
            tag string
        seed (:obj:`int`, *optional*, defaults to 0):
            rng seed for reproductibility

    return: str
    """

    tags = tags.split(" ")
    random.Random(len(tags)+seed).shuffle(tags)
    tags = " ".join(tags)
    return(tags)
