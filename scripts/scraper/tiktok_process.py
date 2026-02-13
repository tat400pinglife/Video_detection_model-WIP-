"""
This is the .py version of tiktok_link.ipynb.
tiktok_lib.py is a modified version of the pyktok library.

You MUST collect the links first into a folder.
Then, you can use save_video_batch to download a video, pass it to a function, and delete the video.
This is because TikTok limits how many videos you can see, therefore how many links you can grab.
Please call the link grabber repeatedly to build urls.csv.
"""

from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from typing import Callable

import pandas as pd
import time
import re
import os

import tiktok_lib as tt

options = Options()
service = Service(ChromeDriverManager().install())

def grab_tiktok_links(
        url: str = 'https://www.tiktok.com/tag/ai',
        a_class: str = 'AMetaCaptionLine',
        wait_time: int = 1,
        max_tries: int = 3
        ) -> list[str]:
    """
    Args:
        url: Tiktok hashtag url
        a_class: <a> container class for lookup
        wait_time: Seconds to pause between checks. Increase if poor internet
        max_tries: Number of failed scrolls before exiting

    Returns:
        links: Unique urls. Size not match goal.
    """
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    wait = WebDriverWait(driver, timeout=5)
    html = driver.find_element(By.TAG_NAME,"html")
    links = set()
    
    a_selector = f"a[class*='{a_class}']"
    count = 0
    wait.until(
        lambda d: len(d.find_elements(By.CSS_SELECTOR, a_selector)) != 0
    )
    prev_elements = []
    try:
        while count <= max_tries:
            elements = driver.find_elements(By.CSS_SELECTOR, a_selector)

            for el in elements:
                href = el.get_attribute("href")
                if href:
                    links.add(href.split("?")[0])

            html.send_keys(Keys.END)
            
            time.sleep(wait_time)

            if elements == prev_elements:
                count += 1
            else:
                count = 0
            prev_elements = elements

    finally:
        driver.quit()

    print(f"Successfully grabbed {len(links)} urls.")
    
    return list(links)




def save_links(
        links: list,
        path: Path = Path("csvs"),
        filename: str = "urls.csv"
        ) -> None:
    """
    Args:
        path: Defaults to "csvs" folder in same dir
    """
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename

    new_df = pd.DataFrame(links, columns=["url"])

    if not filepath.exists():
        new_df.to_csv(filepath, index=False)
    else:
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["url"], keep="first")
        combined_df.to_csv(filepath, index=False)



def save_video_from_url(
        url: str,
        path: Path = Path("data")
        ) -> Path:
    """
    Args:
        url: TikTok video url
        path: Defaults to "data" folder in same dir
    Returns:
        Path to video
    """
    path.mkdir(parents=True, exist_ok=True)

    # Example URL: https://www.tiktok.com/@username/video/7589040432898657550

    username_match = re.search(r'@([^/]+)', url)
    username = username_match.group(1) if username_match else None

    video_id_match = re.search(r'/video/(\d+)', url)
    video_id = video_id_match.group(1) if video_id_match else None

    video_fn = path / f"@{username}_{video_id}.mp4"

    tt.save_tiktok(url, True,
                   video_fn = video_fn,
                   metadata_fn = path / 'metadata.csv'
                   )
    return video_fn


def save_video_batch(
        links: list[str] = [],
        link_folder: Path = Path('csvs'),
        start: int = 0,
        goal: int = 10,
        wait: int = 0,
        path: Path = Path("data"),
        fn: Callable = None,
        delete_after: bool = True,
        *args, **kwargs
        ) -> None:
    
    """
    From the urls.csv, this function will download videos from index [start:start + goal]. If provided a fn, it will call the fn with the path to the video as a parameter along with any other arguments provided with *args and **kwargs.

    Args:
        links: List of video URLs. If none given, will search for them
        start: From what index to begin
        goal: How many videos to download before stopping
        wait: Seconds to wait between downloads
        path: Defaults to "data" folder in same dir
        fn: Function to run
        delete_after: If fn is provided, mp4 will be deleted after running the fn
        *args, **kwargs: Args to pass onto the fn
    """

    if not links:
        csv_files = list(link_folder.glob('*.csv'))
        
        if csv_files:
            dfs = [pd.read_csv(file) for file in csv_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            
            links = combined_df['url'].tolist()
            print(f"Loaded {len(links)} URLs from {len(csv_files)} CSV files")
        else:
            print(f"No CSV files found in {link_folder}")

    if not links:
        print("No links found, and none could be imported.")
        return
    if start > len(links):
        print(f"Start index {start} is Out of range.")
        return
    end = min(len(links), start + goal)

    for i in tqdm(range(start, end), 
              desc="Downloading videos",
              unit="video"):
        res_path = save_video_from_url(links[i], path)
        if fn != None:
            fn(res_path, *args, **kwargs)
            if delete_after:
                os.remove(res_path)

        time.sleep(wait)
    print(f"Videos saved to {path}.")


if __name__ == "__main__":
    res = grab_tiktok_links()
    save_links(res)

    """
    Here is an example of how to use save_video_batch:
    def dummy(path):
        print("Dummy function test:", path)

    save_video_batch(goal = 1,
                    fn = dummy
                    )
    """