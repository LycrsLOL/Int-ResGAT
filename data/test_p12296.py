import re
with open(r'c:\Users\h2406\Desktop\Int_ResGAT\data\sheet1.xml', 'r', encoding='utf-8') as f:
    content = f.read()
    row_start = content.find('<c r="A3850" t="inlineStr"><is><t>P12296</t></is></c>')
    if row_start != -1:
        row_end = content.find('</row>', row_start)
        row_text = content[row_start:row_end]
        matches = re.finditer(r'<c r="([A-Z]+)\d+"[^>]*><is><t>(.*?)</t></is></c>', row_text)
        for m in matches:
            print(f'{m.group(1)}: {m.group(2)}')
