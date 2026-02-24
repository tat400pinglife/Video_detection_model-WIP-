# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:06:01 2022

@author: freelon
"""

import asyncio
import browser_cookie3
from bs4 import BeautifulSoup
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
import random
import re
import requests
from TikTokApi import TikTokApi
import time

global cookies
cookies = dict()

url_regex = '(?<=\.com/)(.+?)(?=\?|$)'
video_id_regex = '(?<=/video/)([0-9]+)'

ms_token = os.environ.get(
    "ms_token", None
)

headers = {'Accept-Encoding': 'gzip, deflate, sdch',
           'Accept-Language': 'en-US,en;q=0.8',
           'Upgrade-Insecure-Requests': '1',
           'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
           'Cache-Control': 'max-age=0',
           'Connection': 'keep-alive'}
context_dict = {'viewport': {'width': 0,
                             'height': 0},
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'}

runsb_rec = ('If pyktok does not operate as expected, you may find it helpful to run the \'specify_browser\' function. \'specify_browser\' takes as its sole argument a string representing a browser installed on your system, e.g. "chrome," "firefox," "edge," etc.')
runsb_err = 'No browser defined for cookie extraction. We strongly recommend you run \'specify_browser\', which takes as its sole argument a string representing a browser installed on your system, e.g. "chrome," "firefox," "edge," etc.'

print(runsb_rec)

class BrowserNotSpecifiedError(Exception):
    def __init__(self):
        super().__init__(runsb_err)

def specify_browser(browser):
    global cookies
    cookies = getattr(browser_cookie3,browser)(domain_name='.tiktok.com')

def deduplicate_metadata(metadata_fn,video_df,dedup_field='video_id'):
    if os.path.exists(metadata_fn):
        metadata = pd.read_csv(metadata_fn,keep_default_na=False)
        combined_data = pd.concat([metadata,video_df])
        combined_data[dedup_field] = combined_data[dedup_field].astype(str)
    else:
        combined_data = video_df
    return combined_data.drop_duplicates(dedup_field)

def generate_data_row(video_obj):
    data_header = ['video_id',
                   'video_timestamp',
                   'video_duration',
                   'video_locationcreated',
                   'video_diggcount',
                   'video_sharecount',
                   'video_commentcount',
                   'video_playcount',
                   'video_description',
                   'video_is_ad',
                   'video_stickers',
                   'author_username',
                   'author_name',
                   'author_followercount',
                   'author_followingcount',
                   'author_heartcount',
                   'author_videocount',
                   'author_diggcount',
                   'author_verified',
                   'poi_name',
                   'poi_address',
                   'poi_city']
    data_list = []
    data_list.append(video_obj['id'])
    try:
        ctime = video_obj['createTime']
        data_list.append(datetime.fromtimestamp(int(ctime)).isoformat())
    except Exception:
        data_list.append('')
    try:
        data_list.append(video_obj['video']['duration'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['locationCreated'])
    except Exception:
        data_list.append('')
    try:
        data_list.append(video_obj['stats']['diggCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['stats']['shareCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['stats']['commentCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['stats']['playCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['desc'])
    except Exception:
        data_list.append('')
    try:
        data_list.append(video_obj['isAd'])
    except Exception:
        data_list.append(False)
    try:
        video_stickers = []
        for sticker in video_obj['stickersOnItem']:
            for text in sticker['stickerText']:
                video_stickers.append(text)
        data_list.append(';'.join(video_stickers))
    except Exception:
        data_list.append('')
    try:
        data_list.append(video_obj['author']['uniqueId'])
    except Exception:
        try:
            data_list.append(video_obj['author'])
        except Exception:
            data_list.append('')
    try:
        data_list.append(video_obj['author']['nickname'])
    except Exception:
        try:
            data_list.append(video_obj['nickname'])
        except Exception:
            data_list.append('')
    try:
        data_list.append(video_obj['authorStats']['followerCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['authorStats']['followingCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['authorStats']['heartCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['authorStats']['videoCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['authorStats']['diggCount'])
    except Exception:
        data_list.append(np.nan)
    try:
        data_list.append(video_obj['author']['verified'])
    except Exception:
        data_list.append(False)
    try:
        data_list.append(video_obj['poi']['name'])
    except Exception:
        data_list.append('')
    try:
        data_list.append(video_obj['poi']['address'])
    except Exception:
        data_list.append('')
    try:
        data_list.append(video_obj['poi']['city'])
    except Exception:
        data_list.append('')
    data_row = pd.DataFrame(dict(zip(data_header,data_list)),index=[0])
    return data_row
#currently unused, but leaving it in case it's needed later
'''
def fix_tt_url(tt_url):
    if 'www.' not in tt_url.lower():
        url_parts = tt_url.split('://')
        fixed_url = url_parts[0] + '://www.' + url_parts[1]
        return fixed_url
    else:
        return tt_url
'''
def get_tiktok_json(video_url,browser_name=None):
    if 'cookies' not in globals() and browser_name is None:
        raise BrowserNotSpecifiedError
    global cookies
    if browser_name is not None:
        cookies = getattr(browser_cookie3,browser_name)(domain_name='.tiktok.com')
    tt = requests.get(video_url,
                      headers=headers,
                      cookies=cookies,
                      timeout=20)
    # retain any new cookies that got set in this request
    cookies = tt.cookies
    soup = BeautifulSoup(tt.text, "html.parser")
    tt_script = soup.find('script', attrs={'id':"SIGI_STATE"})
    try:
        tt_json = json.loads(tt_script.string)
    except AttributeError:
        return
    return tt_json

def alt_get_tiktok_json(video_url,browser_name=None):
    if 'cookies' not in globals() and browser_name is None:
        raise BrowserNotSpecifiedError
    global cookies
    if browser_name is not None:
        cookies = getattr(browser_cookie3,browser_name)(domain_name='.tiktok.com')
    tt = requests.get(video_url,
                      headers=headers,
                      cookies=cookies,
                      timeout=20)
    # retain any new cookies that got set in this request
    cookies = tt.cookies
    soup = BeautifulSoup(tt.text, "html.parser")
    tt_script = soup.find('script', attrs={'id':"__UNIVERSAL_DATA_FOR_REHYDRATION__"})
    try:
        tt_json = json.loads(tt_script.string)
    except AttributeError:
        print("The function encountered a downstream error and did not deliver any data, which happens periodically for various reasons. Please try again later.")
        return
    return tt_json


def save_tiktok(video_url,
                save_video=False,
                metadata_fn='',
                video_fn=None,
                browser_name=None,
                return_fns=False,
                verbose=False):
    """
    Robust save_tiktok with canonical video_id deduping.
    Writes metadata CSV (if metadata_fn) and optionally saves the video file(s).
    Returns {'video_fn': video_fn, 'metadata_fn': metadata_fn} if return_fns True.
    """
    if 'cookies' not in globals() and browser_name is None:
        raise BrowserNotSpecifiedError

    if save_video is False and metadata_fn == '':
        print('Since save_video and metadata_fn are both False/blank, the program did nothing.')
        return

    global cookies
    if browser_name is not None:
        cookies = getattr(browser_cookie3, browser_name)(domain_name='.tiktok.com')

    # Try primary JSON extraction first
    tt_json = get_tiktok_json(video_url, browser_name)

    # Fallback to alt JSON if necessary
    if tt_json is None:
        tt_json = alt_get_tiktok_json(video_url, browser_name)
        if tt_json is None:
            if verbose:
                print("Failed to retrieve tiktok JSON for:", video_url)
            return

    # --- Normalize/extract the item data slot and video_id from likely structures ---
    data_slot = None
    video_id = None
    try:
        # Primary structure
        if isinstance(tt_json, dict) and 'ItemModule' in tt_json and tt_json['ItemModule']:
            video_id = list(tt_json['ItemModule'].keys())[0]
            data_slot = tt_json['ItemModule'][video_id]
        # Alternative structure under __DEFAULT_SCOPE__
        elif isinstance(tt_json, dict) and "__DEFAULT_SCOPE__" in tt_json:
            default = tt_json["__DEFAULT_SCOPE__"]
            # common nested path used in the alt parser
            data_slot = (default.get('webapp.video-detail', {})
                              .get('itemInfo', {})
                              .get('itemStruct', {}))
            # attempt to find id in that structure
            video_id = data_slot.get('id') or data_slot.get('itemId') or data_slot.get('item_id')
        # Last resort: try to parse id from the URL
        if not video_id:
            vid_matches = re.findall(video_id_regex, video_url)
            if vid_matches:
                video_id = vid_matches[0]
        # If data_slot still empty, try a best-effort assignment
        if data_slot is None:
            # attempt to pick a top-level item module entry if present
            if isinstance(tt_json, dict) and 'ItemModule' in tt_json and tt_json['ItemModule']:
                data_slot = list(tt_json['ItemModule'].values())[0]
            else:
                data_slot = tt_json  # last resort
    except Exception:
        # fallback - extract id from url and set data_slot to tt_json
        try:
            video_id = re.findall(video_id_regex, video_url)[0]
        except Exception:
            video_id = None
        data_slot = tt_json

    # --- Generate the data row and ensure video_id is canonical string ---
    data_row = generate_data_row(data_slot)
    # Guarantee the video_id column exists and is canonicalized
    if 'video_id' not in data_row.columns:
        data_row.loc[:, 'video_id'] = str(video_id) if video_id is not None else ''
    else:
        data_row.loc[:, 'video_id'] = data_row.loc[:, 'video_id'].astype(str).str.strip()

    # Try to set author_verified if present in alternate structures (best-effort)
    try:
        if 'UserModule' in tt_json:
            user_id = list(tt_json['UserModule']['users'].keys())[0]
            data_row.loc[0, "author_verified"] = tt_json['UserModule']['users'][user_id].get('verified', data_row.loc[0, "author_verified"])
    except Exception:
        pass

    # --- Save the video (imagePost slides vs single mp4) ---
    out_video_fn = video_fn  # ensure returning the correct value later
    if save_video:
        try:
            # Build filename default from the url regex (same behavior as original)
            regex_url = re.findall(url_regex, video_url)[0] if re.search(url_regex, video_url) else str(video_id)

            # imagePost (slides)
            if isinstance(data_slot, dict) and 'imagePost' in data_slot:
                slidecount = 1
                for slide in data_slot['imagePost'].get('images', []):
                    if video_fn:
                        base, ext = os.path.splitext(video_fn)
                        slide_fn = f"{base}_slide_{slidecount}.jpeg"
                    else:
                        slide_fn = regex_url.replace('/', '_') + f'_slide_{slidecount}.jpeg'

                    tt_video_url = slide.get('imageURL', {}).get('urlList', [None])[0]
                    if not tt_video_url:
                        continue
                    headers['referer'] = 'https://www.tiktok.com/'
                    tt_video = requests.get(tt_video_url, allow_redirects=True, headers=headers, cookies=cookies, timeout=30)
                    with open(slide_fn, 'wb') as fn:
                        fn.write(tt_video.content)
                    if verbose:
                        print("Saved image slide:", slide_fn)
                    slidecount += 1
                # set out_video_fn to the first slide or the provided name if given
                if not video_fn:
                    out_video_fn = regex_url.replace('/', '_') + '_slide_1.jpeg'
            else:
                # mp4 video case
                if video_fn:
                    out_fn = video_fn
                else:
                    out_fn = (regex_url.replace('/', '_') + '.mp4') if 'regex_url' in locals() else f'{video_id}.mp4'

                # try a couple of known paths to the download/play address
                tt_video_url = None
                try:
                    # primary structure
                    tt_video_url = (data_slot.get('video', {})
                                            .get('downloadAddr'))
                except Exception:
                    tt_video_url = None

                if not tt_video_url:
                    # fallback nested path used in the alt JSON
                    try:
                        tt_video_url = tt_json["__DEFAULT_SCOPE__"]['webapp.video-detail']['itemInfo']['itemStruct']['video'].get('downloadAddr')
                    except Exception:
                        tt_video_url = None

                if not tt_video_url:
                    try:
                        tt_video_url = data_slot.get('video', {}).get('playAddr')
                    except Exception:
                        tt_video_url = None

                if not tt_video_url:
                    # final fallback: try scanning tt_json for a likely video url (best-effort)
                    # (we avoid aggressive search; if not found, raise)
                    raise ValueError("Could not find a downloadable video URL in the JSON.")

                headers['referer'] = 'https://www.tiktok.com/'
                tt_video = requests.get(tt_video_url, allow_redirects=True, headers=headers, cookies=cookies, timeout=60)
                with open(out_fn, 'wb') as fn:
                    fn.write(tt_video.content)
                out_video_fn = out_fn
                if verbose:
                    print("Saved video\n", tt_video_url, "\nto\n", os.path.abspath(out_fn))
        except Exception as e:
            # If saving the video fails, continue — metadata can still be written
            if verbose:
                print("Warning: failed to save video for", video_url, ":", repr(e))

    # --- Write metadata with canonical dedupe on video_id ---
    if metadata_fn:
        try:
            if os.path.exists(metadata_fn):
                metadata = pd.read_csv(metadata_fn, keep_default_na=False)
                combined_data = pd.concat([metadata, data_row], ignore_index=True)
            else:
                combined_data = data_row.copy()

            # canonicalize dedupe key
            combined_data['video_id'] = combined_data['video_id'].astype(str).str.strip()

            # dedupe and keep the latest (the newly-added row)
            combined_data = combined_data.drop_duplicates(subset=['video_id'], keep='last').reset_index(drop=True)

            combined_data.to_csv(metadata_fn, index=False)
            if verbose:
                print("Saved metadata for video\n", video_url, "\nto\n", os.path.abspath(metadata_fn))
        except Exception as e:
            print("Error saving metadata:", repr(e))

    if return_fns:
        return {'video_fn': out_video_fn, 'metadata_fn': metadata_fn}


# the function below is based on this one: https://github.com/davidteather/TikTok-Api/blob/main/examples/user_example.py

async def get_video_urls(tt_ent,
                         ent_type="user",
                         video_ct=30,
                         headless=True):
    if ent_type not in ['user','hashtag','video_related']:
        raise Exception('Only allowed `ent_type` values are "user", "hashtag", or "video_related".')

    url_p1 = "https://www.tiktok.com/@"
    url_p2 = "/video/"
    tt_list = []

    async with TikTokApi() as api:
        await api.create_sessions(headless=headless,
                                  ms_tokens=[ms_token],
                                  num_sessions=1,
                                  sleep_after=3,
                                  context_options=context_dict)
        if ent_type == 'user':
            ent = api.user(tt_ent)
        elif ent_type == 'hashtag':
            ent = api.hashtag(name=tt_ent)
        else:
            ent = api.video(url=tt_ent)

        if ent_type in ['user','hashtag']:
            async for video in ent.videos(count=video_ct):
                tt_list.append(video.as_dict)
        else:
            async for related_video in ent.related_videos(count=video_ct):
                tt_list.append(related_video.as_dict)

    id_list = [i['id'] for i in tt_list]
    if ent_type == 'user':
        video_list = [url_p1 + tt_ent + url_p2 + i for i in id_list]
    else:
        author_list = [i['author']['uniqueId'] for i in tt_list]
        video_list = []
        for n, i in enumerate(author_list):
            video_url = url_p1 + author_list[n] + url_p2 + id_list[n]
            video_list.append(video_url)
    return video_list[:video_ct]

def save_tiktok_multi_urls(video_urls,
                           save_video=False,
                           metadata_fn='',
                           sleep=4,
                           browser_name=None):
    if 'cookies' not in globals() and browser_name is None:
        raise BrowserNotSpecifiedError
    if type(video_urls) is str:
        tt_urls = open(video_urls).read().splitlines()
    else:
        tt_urls = video_urls
    for u in tt_urls:
        save_tiktok(u,save_video,metadata_fn,browser_name)
        time.sleep(random.randint(1, sleep))
    print('Saved',len(tt_urls),'videos and/or lines of metadata')

def save_tiktok_multi_page(tt_ent,
                           ent_type="user",
                           video_ct=30,
                           headless=True,
                           save_video=False,
                           metadata_fn='',
                           sleep=4,
                           browser_name=None):
    video_urls = asyncio.run(get_video_urls(tt_ent,
                                            ent_type,
                                            video_ct,
                                            headless))
    save_tiktok_multi_urls(video_urls,
                           save_video,
                           metadata_fn,
                           sleep,
                           browser_name)

# the function below is based on this one: https://github.com/davidteather/TikTok-Api/blob/main/examples/comment_example.py

async def get_comments(video_id,comment_count=30,headless=True):
    comment_list = []
    async with TikTokApi() as api:
        await api.create_sessions(headless=headless,
                                  ms_tokens=[ms_token],
                                  num_sessions=1,
                                  sleep_after=3,
                                  context_options=context_dict)
        video = api.video(id=video_id)
        async for comment in video.comments(count=comment_count):
            comment_list.append(comment.as_dict)
    return pd.DataFrame(comment_list)

def save_tiktok_comments(video_url,
                         filename='',
                         comment_count=30,
                         headless=True,
                         save_comments=True,
                         return_comments=True):
    video_id = int(re.findall(video_id_regex,video_url)[0])
    comment_results = asyncio.run(get_comments(video_id,comment_count,headless))
    if save_comments:
        if filename == '':
            regex_url = re.findall(url_regex, video_url)[0]
            filename = regex_url.replace('/', '_') + '_comments.csv'
        data_to_save = deduplicate_metadata(filename,comment_results,'cid')
        data_to_save.to_csv(filename,mode='w',index=False)
        print(len(comment_results),"comments saved.")
    if return_comments:
        return comment_results
