import datetime

import bson
import pytest


@pytest.fixture
def tiny_bson(tmp_path):
    docs = [
        {"_id": 1, "Title": "Վերնագիր մեկ", "Text": "տեքստ " * 40,
         "Source": "armenpress.am", "PostDate": datetime.datetime(2025, 5, 1),
         "ScrapeDate": datetime.datetime(2025, 5, 2), "Author": "Ա. Բ.",
         "Href": "https://armenpress.am/arm/news/1?utm_source=fb", "Topic": "Քաղաքական"},
        {"_id": 2, "Title": "", "Text": "no title " * 40},                      # dropped
        {"_id": 3, "Title": "Կարճ", "Text": "x"},                               # dropped (<100)
        {"_id": 4, "Title": "Երկրորդ", "Text": "բովանդակություն " * 30,
         "Source": "news.am", "PostDate": None},
    ]
    p = tmp_path / "tiny.bson"
    with open(p, "wb") as f:
        for d in docs:
            f.write(bson.encode(d))
    return p
