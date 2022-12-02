## Data preparation
1. Prepare the datasets with the [download_data.sh](../data/download_data.sh) script.

    ```bash
    bash data/download_data.sh --path ./data
    ```
    You can also download these datasets manually using the URLs in the [download_data.sh](../data/download_data.sh) script. We have provided some alternative download links [here](https://drive.google.com/drive/folders/1-faf4GiPBTwzEItdphhjlIZlP30GIPoc?usp=sharing), as the download URLs for some files are not functioning. 

2. Download the dataset index files from [Google Drive](https://drive.google.com/file/d/1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS/view?usp=sharing) 
to the ``split/`` folder and then extract them.
    ```bash
    cd split
    tar -xf data.tar
    ```

    The folder structure for these datasets is shown below.

    ```text
    VLTVG
    ├── data
    │   ├── Flickr30k
    │   │   ├── flickr30k-images
    │   ├── other
    │   │   ├── images
    │   ├── referit
    │   │   ├── images
    │   │   ├── masks
    ├── split
    │   ├── data
    │   │   ├── flickr
    │   │   ├── gref
    │   │   ├── gref_umd
    │   │   ├── referit
    │   │   ├── unc
    │   │   ├── unc+    
    ```
    
## Pretrained Checkpoints
We use the pretrained checkpoints provided by [TransVG](https://github.com/djiajunustc/TransVG). 
You can download them [here](https://drive.google.com/drive/folders/1SOHPCCR6yElQmVp96LGJhfTP46RxVwzF?usp=sharing) 
and place them in the ``pretrained_checkpoints/`` folder.
