import re

text = '李明在华为公司工作，他住在北京市。昨天去了腾讯大厦。'

# 测试负向前瞻模式
suffix = '公司'
pattern = rf'(?<![\u4e00-\u9fa5])([\u4e00-\u9fa5]{{2,4}}){suffix}(?![\u4e00-\u9fa5])'
matches = list(re.finditer(pattern, text))
print(f'负向前瞻模式: {len(matches)} 个匹配')
for m in matches:
    print(f'  - {m.group()} at {m.start()}-{m.end()}')

# 检查为什么匹配不到
print(f'\n检查 "华为公司" 的位置:')
idx = text.find('华为公司')
print(f'  起始位置: {idx}')
if idx > 0:
    prev_char = text[idx-1]
    is_chinese = '\u4e00' <= prev_char <= '\u9fa5'
    print(f'  前一个字符: "{prev_char}" (是否是中文: {is_chinese})')

# 测试更简单的模式
print(f'\n测试简单模式:')
simple_pattern = r'华为公司'
simple_matches = list(re.finditer(simple_pattern, text))
print(f'  简单模式匹配: {len(simple_matches)} 个')
