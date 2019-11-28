import urllib.request as rq
from bs4 import BeautifulSoup as bs
import re

nips_base = r'https://papers.nips.cc'
def find_by_re(url, pattern, out_file = "out.md"):
    url_content = rq.urlopen(url)
    soup = bs(url_content, 'html.parser')
    pattern = re.compile(pattern)
    res = soup.find_all('a', text = pattern)
    with open(out_file, "w") as f:
        f.write("# Search Result\n")
        for it in res:
            f.write('- ' + it.text + "\n")
            fa = it.parent
            f.write(r'- Authors: ')
            fa_res = fa.find_all(class_='author')
            for iter in fa_res:
                f.write(iter.text + '; ')
            f.write('\n> [url](%s%s)\n\n' % (nips_base, it['href']))

