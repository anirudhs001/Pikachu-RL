
import time
import hashlib
from tkinter import W
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
import pytz
import notify2


last_seen_num_items = 0

while True:
    page = urlopen('https://shop.amul.in/WebForms/Web_Dist_Category_PrdList.aspx?DistId=MTExMTExMQ==&PCatId=MQ==&IsDistPrd=VHJ1ZQ==#')
    soup = BeautifulSoup(page, "html.parser")
    items = []
    for product in soup.findAll("td", class_="dxdvItem"):
        prod_label = product.find('div', class_='product-content').find("h3", class_="title").find('span').get('title').replace(":", "")
        # prod_label = " ".join(prod_label)
        out_of_stock_div = product.find('span', class_='product-outofstock-label')
        if out_of_stock_div is None:
            items.append(prod_label)

    datetime_india = datetime.now(pytz.timezone('Asia/Kolkata'))
    s = datetime_india.strftime("%a, %d %b %H:%M:%S")
    print('\r' + s + " " + "|".join(items), end="")
    # s = datetime_india.strftime("%a, %d %b %H:%M:%S", time.gmtime(time.time()))
    if last_seen_num_items != len(items):
        print("\n[NEW ITEM IN STOCK!!]")
        notify2.init("Test")
        n = notify2.Notification(
            f"{len(items)} Item(s) In stock", 
            "\n".join(items),
        )
        n.show()
        last_seen_num_items = len(items)
    time.sleep(60)
            
