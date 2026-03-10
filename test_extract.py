from skills.graphrag.services import HybridEntityExtractor
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    extractor = HybridEntityExtractor(persist_dir=tmpdir)
    
    # 测试 extract_from_query
    query = '肺气肿怎么治疗'
    keywords = extractor.extract_from_query(query)
    
    print(f'查询: {query}')
    print(f'提取关键词: {len(keywords)} 个')
    for k in keywords:
        print(f"  - {k['name']} ({k['type']})")
