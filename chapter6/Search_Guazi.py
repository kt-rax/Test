# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:17:56 2019

@author: hefug
"""

import requests
from requests import RequestException
from bs4 import BeautifulSoup as bs
import time
import re
import pymongo


class GuaZi:
    def __init__(self):
        self.link = 'https://www.guazi.com/cd/buy/o'
        self.headers = {
                'Cookie': 'uuid=8e189c5e-4b3c-4eca-9f69-50d11cd70f62; ganji_uuid=5614770255330340838852; lg=1; antipas=L693382z8954211H66291335Huw3; clueSourceCode=10104346512%2300; sessionid=f340bcee-4390-4aab-e05d-4273c453d102; cainfo=%7B%22ca_s%22%3A%22dh_hao123llq%22%2C%22ca_n%22%3A%22hao123mzpc%22%2C%22ca_i%22%3A%22-%22%2C%22ca_medium%22%3A%22-%22%2C%22ca_term%22%3A%22-%22%2C%22ca_content%22%3A%22-%22%2C%22ca_campaign%22%3A%22-%22%2C%22ca_kw%22%3A%22-%22%2C%22keyword%22%3A%22-%22%2C%22ca_keywordid%22%3A%22-%22%2C%22scode%22%3A%2210104346512%22%2C%22ca_transid%22%3Anull%2C%22platform%22%3A%221%22%2C%22version%22%3A1%2C%22ca_b%22%3A%22-%22%2C%22ca_a%22%3A%22-%22%2C%22display_finance_flag%22%3A%22-%22%2C%22client_ab%22%3A%22-%22%2C%22guid%22%3A%228e189c5e-4b3c-4eca-9f69-50d11cd70f62%22%2C%22sessionid%22%3A%22f340bcee-4390-4aab-e05d-4273c453d102%22%7D; cityDomain=cd; preTime=%7B%22last%22%3A1545284609%2C%22this%22%3A1544171667%2C%22pre%22%3A1544171667%7D',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'
                }
        self.client = pymongo.MongoClient(host='127.0.0.1', port=27017)
        self.db = self.client.spider        

    def get_page(self, url):
        try:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status
            resp.encoding = resp.apparent_encoding
            return resp.text
        except RequestException:
            print('Can not get the page')
            pass
    
    def get_link(self, html):
        #soup = bs(html, 'lxml')
        #changed by KT 2021-Oct-01 fix TypeError: object of type 'NoneType' has no len() ref: https://stackoverflow.com/questions/21956956/how-to-load-a-beautifulsoup-page-parser
        soup = bs(html, 'lxml')
        result = soup.select('div.list-wrap.js-post > ul > li > a')
        detail_url = ['https://www.guazi.com'+i['href'] for i in result]
        return detail_url
    
    def get_detail(self, html):
        detail = bs(html, 'lxml')
        title = detail.select_one('div.infor-main.clearfix > div.product-textbox > h2').get_text()
        title = re.sub(r'[\r\n]', '', title)
        time = detail.select_one('div.product-textbox > ul > li.one > span').get_text()
        used_distance = detail.select_one('div.product-textbox > ul > li.two > span').get_text()
        city = detail.select('div.product-textbox > ul > li.three > span')[0].get_text()
        displacement = detail.select('div.product-textbox > ul > li.three > span')[1].get_text()
        transmission = detail.select_one('div.product-textbox > ul > li.last > span').get_text()
        price = detail.select_one('div.product-textbox > div.pricebox.js-disprice > span.pricestype').get_text()
        guiding_price = detail.select_one('div.product-textbox > div.pricebox.js-disprice > span.newcarprice').get_text()
        guiding_price = re.sub(r'[\r\n ]', '', guiding_price)
        
        result={
                'title': title.strip().replace('                    ', ' '),
                'time': time,
                'used_distance': used_distance,
                'city': city,
                'displacement':displacement,
                'transmission': transmission,
                'price': price.replace(' ', ''),
                'guiding_price':guiding_price
                }
        return result
    
    def save_to_mongo(self, content):
        if content:
            self.db.ershouche_cd.insert(content)
            print(content['title'], 'DONE')
    
    def main(self):        
        for i in range(1, 101, 1):
            url = self.link+str(i)
            html = self.get_page(url)
            result  =self.get_link(html)
            for i in result:
                time.sleep(2)
                resp = self.get_page(i)
                content = self.get_detail(resp)
                if content:
                    self.save_to_mongo(content)
        

if __name__ == '__main__':
    ershouche = GuaZi()
    ershouche.main()
