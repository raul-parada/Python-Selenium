from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pytest

@pytest.fixture(scope='module')
def driver():
    with webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install())) as driver:
        yield driver

@pytest.mark.parametrize("url", [
    "https://www.wikipedia.org/",
    "https://www.google.com/",
    "https://www.github.com/"
])
def test_multiple_urls(driver, url):
    # Navigate to the URL
    driver.get(url)

    # Wait for the search field to be visible
    wait = WebDriverWait(driver, 10)
    #elem = wait.until(EC.visibility_of_element_located((By.NAME, "search")))

    cookies = driver.get_cookies()
    print(cookies)
