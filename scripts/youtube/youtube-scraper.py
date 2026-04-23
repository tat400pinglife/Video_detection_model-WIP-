"""
This works the exact same way as the tiktok one.
If you want to pull URLs yourself, you need to have your own API key.
"""


from typing import Dict, List, Tuple, Callable
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import re
import os
import time
import pandas as pd
import numpy as np
import yt_dlp

YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
SHORTS_URL_TEMPLATE = "https://www.youtube.com/shorts/{}"

ISO8601_RE = re.compile(
    r'^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$'
)

load_dotenv()
API_KEY = os.getenv('API_KEY')


def save_links(
        links: list,
        path: Path = Path("csvs"),
        filename: str = "youtube.csv"
        ) -> None:
    """
    Args:
        path: Defaults to "csvs" folder in same dir
    """
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename

    new_df = pd.DataFrame({
        "url": links,
        "valid": np.nan
    })

    if not filepath.exists():
        new_df.to_csv(filepath, index=False)
    else:
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["url"], keep="first")
        combined_df.to_csv(filepath, index=False)


def iso8601_to_seconds(dur: str) -> int:
    """
    Convert ISO 8601 to seconds. Used for checking vid length.
    Args:
        dur: iso8601 string
    Returns:
        secs: duration as a number 
    """
    if not dur:
        return -1
    m = ISO8601_RE.match(dur)
    if not m:
        return -1
    parts = {k: int(v) if v else 0 for k, v in m.groupdict().items()}
    return parts['days'] * 86400 + parts['hours'] * 3600 + parts['minutes'] * 60 + parts['seconds']


def shorts_url_check(video_id: str,
                     timeout: float = 6.0
                     ) -> bool:
    """
    HEAD-check the /shorts/{id} URL.
    200 likely = actual Shorts page;
    redirect (~303) likely = normal watch page.
    This may be unreliable.
    """
    url = SHORTS_URL_TEMPLATE.format(video_id)
    try:
        # use HEAD first; fall back to GET if server doesn't respond to HEAD
        r = requests.head(url, allow_redirects=False, timeout=timeout)
        code = r.status_code
        if code == 200:
            return True
        if code in (301, 302, 303, 307, 308):
            # redirected: probably not a Shorts "shorts" page
            return False
        # fallback: try GET but do not follow redirects
        r2 = requests.get(url, allow_redirects=False, timeout=timeout)
        return r2.status_code == 200
    except Exception:
        return False


def build_shorts_urls(video_ids: List[str]) -> List[str]:
    return [SHORTS_URL_TEMPLATE.format(v) for v in video_ids]


def fetch_durations(api_key: str,
                    video_ids: List[str]
                    ) -> Dict[str, int]:
    """
    Batch-call videos.list (part=contentDetails) for given IDs (batches of 50).
    Returns:
        res: hmap of videoId -> duration_seconds (int)
    """
    ids = list(video_ids)
    res = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i+50]
        params = {
            "part": "contentDetails",
            "id": ",".join(batch),
            "key": api_key,
            "maxResults": 50
        }
        r = requests.get(YT_VIDEOS_URL, params=params, timeout=20)
        if not r.ok:
            raise RuntimeError(f"videos.list failed: {r.status_code} {r.text}")
        data = r.json()
        for item in data.get("items", []):
            vid = item.get("id")
            dur_iso = item.get("contentDetails", {}).get("duration", "")
            res[vid] = iso8601_to_seconds(dur_iso)

    return res


def query_youtube(api_key: str,
                  query: str,
                  max_results: int = 500,
                  video_type: str = "short",
                  page_token: str | None = None
                  ) -> List[Dict]:
    """
    Use search.list to collect up to max_results items.
    Args:
        query: string to search
        max_results: videos to return
        video_diration: shorts only or any
    Return:
        items: list of raw search
        nextPageToken: for repeated search
    """
    if max_results <= 0:
        return []
    items = []
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 50,
        "key": api_key
    }
    if video_type and video_type != "any":
        params["videoDuration"] = video_type  # "short" => <4 minutes per API docs
    next_token = page_token
    
    while len(items) < max_results:
        if next_token:
            params["pageToken"] = next_token

        r = requests.get(YT_SEARCH_URL, params=params, timeout=20)
        if not r.ok:
            raise RuntimeError(f"search.list failed: {r.status_code} {r.text}")
        data = r.json()
        page_items = data.get("items", [])
        items.extend(page_items)
        next_token = data.get("nextPageToken")

        if not next_token:
            break

    return items, data.get("nextPageToken")


def extract(search_items: Dict) -> List[Tuple[str, float | None]]:
    """
    From search.list items extract (videoId, thumb h:w ratio or None).
    """
    res = []
    for item in search_items:
        vid = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        thumbnails = snippet.get("thumbnails", {}) or {}
        thumb = thumbnails.get("default")
        if thumb:
            w = thumb.get("width")
            h = thumb.get("height")
            try:
                ratio = float(h) / float(w) if w and h else None
            except Exception:
                ratio = None
        else:
            ratio = None
        if vid:
            res.append((vid, ratio))
    
    return res


def collect_shorts_urls(api_key: str,
                        query: str,
                        goal: int = 500,
                        max_dur: int = 60,
                        min_ratio: float | None = 1.6,
                        use_url_check: bool = False,
                        ) -> List[str]:
    """
    Args:
        max_dur: Maximum vid duration (shorts are <60s)
        min_ratio: minimum thumbnail height/width ratio for vertical video; set None to skip (this doesn't work rn)
        url_check: if True, perform HEAD checks on /shorts/{id} to confirm (watch for rate limits)
    Returns:
        list of video urls
    """

    next_token = None
    filtered = []
    
    while len(filtered) < goal:

        candidates, next_token = query_youtube(
            api_key=api_key,
            query=query,
            max_results=50,
            page_token=next_token
        )

        if not candidates:
            break

        id_ratio_pairs = extract(candidates)
        ids = [vid for vid, _ in id_ratio_pairs]

        durations = fetch_durations(api_key=api_key, video_ids=ids)

        for vid, ratio in id_ratio_pairs:
            dur = durations.get(vid, 1000)

            if dur <= max_dur:
                if min_ratio is None or (ratio and ratio >= min_ratio):
                    filtered.append(vid)

        if not next_token:
            break


    if use_url_check and filtered:
        final = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(shorts_url_check, vid): vid for vid in filtered}
            for fut in as_completed(futures):
                vid = futures[fut]
                try:
                    ok = fut.result()
                except Exception:
                    ok = False
                if ok:
                    final.append(vid)
        filtered = final

    seen = set()
    dedup = []
    for v in filtered:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    return build_shorts_urls(dedup)


def save_video_from_url(
        url: str,
        path: Path = Path("data")
        ) -> Path | None:
    """
    Args:
        url: YouTube video url
        path: Defaults to "data" folder in same dir
    Returns:
        Path to video
    """
    path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(path / "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if "requested_downloads" in info:
                return Path(info["requested_downloads"][0]["filepath"])

            return None

    except Exception:
        return None


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
    From the youtube.csv, this function will download videos from index [start:start + goal]. If provided a fn, it will call the fn with the path to the video as a parameter along with any other arguments provided with *args and **kwargs.

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
            df = pd.concat(dfs, ignore_index=True)
            
            print(f"Loaded {len(df)} URLs from {len(csv_files)} CSV files")
        else:
            print(f"No CSV files found in {link_folder}")

    length = len(df)
    if length == 0:
        print("No links found, and none could be imported.")
        return
    
    if start > length:
        print(f"Start index {start} is Out of range.")
        return
    
    end = min(length, start + goal)
    count = start
    for url, valid in tqdm(df.iloc[start:end].itertuples(index=False, name='Row'),
                    desc="Downloading videos",
                    unit="th video"):
        if valid == 'n' or pd.isna(valid):
            # Not an valid video OR unvalidated, skip
            continue
        
        try:
            res_path = save_video_from_url(url, path)
        except Exception as e:
            print(f"Error on index {count} with {url}: {e}")
            continue 
        if fn != None and res_path != None and os.path.exists(res_path):
            fn(res_path, *args, **kwargs)
            if delete_after:
                os.remove(res_path)
        count += 1
        time.sleep(wait)
    print(f"Finished processing videos from [{start}, {end}).")


if __name__ == "__main__":
    def dummy(path):
        print("Dummy function test:", path)
    save_video_batch(start=0,
                    goal = 1,
                    fn = dummy
                    )
