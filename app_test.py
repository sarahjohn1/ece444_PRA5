import pytest
from pytest_benchmark.plugin import benchmark
import requests
import time
import csv
import pandas as pd
from application import application

URL = "http://server-sentiment-env-2.eba-d3nxfupt.us-east-2.elasticbeanstalk.com/predict"
test_cases = [
    "This is completely false news",  # fake
    "This is a real news story about politics",  # real
    "A fabricated story about the economy",  # fake
    "Joe Biden is president"  # real
]
# Test client for the Flask app
@pytest.fixture
def client():
    with application.test_client() as client:
        yield client

# Test case for fake and real news
def test_fake_news(client):
    response = client.post('/predict', data={'sentence': 'This is completely false news'})
    assert b'This is fake news' in response.data

def test_real_news(client):
    response = client.post('/predict', data={'sentence': 'This is a real news story about politics'})
    assert b'This is real news' in response.data

def test_fake_news_2(client):
    response = client.post('/predict', data={'sentence': 'A fabricated story about the economy'})
    assert b'This is fake news' in response.data

def test_real_news_2(client):
    response = client.post('/predict', data={'sentence': 'Joe Biden is president'})
    assert b'This is real news' in response.data



# Benchmark for fake news test case 1
def test_benchmark_fake_news_1(client, benchmark):
    def post_request():
        return client.post('/predict', data={'sentence': 'This is completely false news'})
    
    result = benchmark(post_request)
    assert b'This is fake news' in result.data

# Benchmark for real news test case 1
def test_benchmark_real_news_1(client, benchmark):
    def post_request():
        return client.post('/predict', data={'sentence': 'This is a real news story about politics'})
    
    result = benchmark(post_request)
    assert b'This is real news' in result.data

# Benchmark for fake news test case 2
def test_benchmark_fake_news_2(client, benchmark):
    def post_request():
        return client.post('/predict', data={'sentence': 'A fabricated story about the economy'})
    
    result = benchmark(post_request)
    assert b'This is fake news' in result.data

# Benchmark for real news test case 2
def test_benchmark_real_news_2(client, benchmark):
    def post_request():
        return client.post('/predict', data={'sentence': 'Joe Biden is president'})
    
    result = benchmark(post_request)
    assert b'This is real news' in result.data

