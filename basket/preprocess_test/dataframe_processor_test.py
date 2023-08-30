import pandas as pd

"""a collection of test function"""

# test suite for resolution_bucketing_batch function
def test_ensure_single_res_each_batch(
    dataframe:pd.DataFrame,
    bucket_group_col_name = "bucket_group",
    image_height_col_name="new_image_height",
    image_width_col_name="new_image_width",
) -> None:

    error_flag = False
    
    min_value = dataframe[bucket_group_col_name].min()
    max_value = dataframe[bucket_group_col_name].max()
    
    for x in range(min_value, max_value):
        width_unique_count = len(dataframe.loc[x][image_width_col_name].unique())
        height_unique_count = len(dataframe.loc[x][image_height_col_name].unique())

        if width_unique_count > 1 or height_unique_count > 1:
            print(
                "FOUND BATCH WITH MULTIPLE RES AT", 
                x, 
                dataframe.loc[x][image_width_col_name].unique(), 
                dataframe.loc[x][image_height_col_name].unique())
            error_flag = True

    if not error_flag:
        print("PASS")
    else:
        print("FAILED")
    pass

def resolution_rescale_test(
    dataframe:pd.DataFrame,
    image_width_col_name:str,
    image_height_col_name:str,
    new_image_width_col_name:str,
    new_image_height_col_name:str,
    threshold:float=0.5
) -> None:
    r"""
    rough test if image cropping exceed threshold (default 0.5)
    for example 4:2 image copped to more than 1:1
    """
    error_flag = False
    # portrait test
    portrait_res = dataframe.loc[dataframe[new_image_height_col_name]/dataframe[new_image_width_col_name]>=1]
    # calculate height/width ratio
    portrait_res_ratio = portrait_res[image_height_col_name]/portrait_res[image_width_col_name]
    # calculate height/width ratio bucket
    new_portrait_res_ratio = portrait_res[new_image_height_col_name]/portrait_res[new_image_width_col_name]
    # calculate scale
    scaled_height = portrait_res[new_image_width_col_name]/portrait_res[image_width_col_name] * portrait_res[image_height_col_name]
    abs_delta = (1-scaled_height/portrait_res[new_image_height_col_name]).abs().max()
    print(abs_delta)
    if abs_delta > threshold:
        print("FAILED")
        error_flag = True

    #landscape test
    landscape_res = dataframe.loc[dataframe[new_image_width_col_name]/dataframe[new_image_height_col_name]>1]
    # calculate height/width ratio
    landscape_res_ratio = landscape_res[image_width_col_name]/landscape_res[image_height_col_name]
    # calculate height/width ratio bucket
    new_landscape_res_ratio = landscape_res[new_image_width_col_name]/landscape_res[new_image_height_col_name]
    # calculate scale
    scaled_width = landscape_res[new_image_height_col_name]/landscape_res[image_height_col_name] * landscape_res[image_width_col_name]
    abs_delta = (1-scaled_width/landscape_res[new_image_width_col_name]).abs().max()
    print(abs_delta)
    if abs_delta > threshold:
        print("FAILED")
        error_flag = True

    if not error_flag:
        print("PASS")
    
    pass