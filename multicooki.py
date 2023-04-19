# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:20:24 2023

@author: cttc
"""

from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import unittest

class TestURLs(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))

    def tearDown(self):
        self.driver.quit()

    def test_multiple_urls(self):
        urls = [
            "https://www.wikipedia.org/",
            "https://www.google.com/",
            "https://www.github.com/"
        ]

        for url in urls:
            # Navigate to the URL
            self.driver.get(url)

            # Wait for the search field to be visible
            wait = WebDriverWait(self.driver, 10)
            #elem = wait.until(EC.visibility_of_element_located((By.NAME, "search")))

            cookies = self.driver.get_cookies()
            print(cookies)
            
if __name__ == '__main__':
    unittest.main()
