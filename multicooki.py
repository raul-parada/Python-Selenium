# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:52:50 2023

@author: cttc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:18:05 2023

@author: cttc
"""
import pytest
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

@pytest.fixture(scope="session")
def driver():
    # Set up the driver
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))
    yield driver
    # Tear down the driver
    driver.quit()

@pytest.mark.parametrize("url", [
    "https://www.wikipedia.org/",
    "https://www.google.com/",
    "https://www.github.com/",
    "https://www.youtube.com/",
    "https://www.facebook.com/",
    "https://www.amazon.com/",
    "https://twitter.com/",
    "https://www.instagram.com/",
    "https://www.linkedin.com/"
    
])
def test_multiple_urls(driver, url):
    # Navigate to the URL
    driver.get(url)

    # Wait for the search field to be visible
    wait = WebDriverWait(driver, 10)
    #elem = wait.until(EC.visibility_of_element_located((By.NAME, "search")))

    # Get the cookies
    cookies = driver.get_cookies()

    # Print a message for each URL
    actual_cookies = [cookie["name"] for cookie in cookies]
    message = f"Received {len(cookies)} cookies for {url}: {actual_cookies}"
    print(message)

    # Assert that we received the expected cookies
    expected_cookies = ['JSESSIONID', 'lang', 'bcookie', 'bscookie', 'li_gc', 'lidc']
    assert set(actual_cookies) == set(expected_cookies), f"Expected {expected_cookies}, but got {actual_cookies} for {url}"

if __name__ == '__main__':
    pytest.main()
