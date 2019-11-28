# Crawler-for-Paper
## Goal
- I'm trying to implement a crawler for retrieving  papers by regular expression in different top conferences.
- I only finished a crawler for nips and more crawlers are coming.
- The input is a regular expression and the corresponding output will be a markdown file which recording related papers.
## How to use
- An example is as follows:
```shell
python main.py --url https://papers.nips.cc/book/advances-in-neural-information-processing-systems-32-2019 --pattern '(g|G)raph'
```
