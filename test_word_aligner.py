"""
WordAligner 集成测试脚本

测试 WordAligner 在 GraphRAG 中的集成效果
"""

import sys
sys.path.insert(0, 'd:\\self_test\\project\\trae\\agent\\OllamaPilot')

from skills.graphrag.word_aligner import (
    WordAligner, 
    AlignedEntity, 
    AlignmentStatus,
    format_alignment_report,
    calculate_chunk_offsets
)
from skills.graphrag.services.entity_extractor import LightweightEntityExtractor


def test_basic_alignment():
    """测试基本对齐功能"""
    print("=" * 70)
    print("测试 1: 基本对齐功能")
    print("=" * 70)
    
    # 原文
    source_text = """太阳病，发热汗出，恶风，脉缓者，名为中风。
太阳病，或已发热，或未发热，必恶寒，体痛，呕逆，脉阴阳俱紧者，名为伤寒。
"""
    
    # 模拟分块
    chunks = [
        "太阳病，发热汗出，恶风，脉缓者，名为中风。",
        "太阳病，或已发热，或未发热，必恶寒，体痛，呕逆，脉阴阳俱紧者，名为伤寒。"
    ]
    
    # 计算偏移量
    chunk_offsets = calculate_chunk_offsets(source_text, chunks)
    print(f"\n📄 原文长度: {len(source_text)} 字符")
    print(f"📦 分块数: {len(chunks)}")
    print(f"📍 块偏移量: {chunk_offsets}")
    
    # 模拟从块中提取的实体
    raw_entities = [
        {'name': '太阳病', 'type': 'MedicalTerm', 'chunk_index': 0, 'start': 0, 'end': 3},
        {'name': '汗出', 'type': 'Symptom', 'chunk_index': 0, 'start': 8, 'end': 10},
        {'name': '中风', 'type': 'Disease', 'chunk_index': 0, 'start': 22, 'end': 24},
        {'name': '太阳病', 'type': 'MedicalTerm', 'chunk_index': 1, 'start': 0, 'end': 3},
        {'name': '恶寒', 'type': 'Symptom', 'chunk_index': 1, 'start': 22, 'end': 24},
        {'name': '伤寒', 'type': 'Disease', 'chunk_index': 1, 'start': 43, 'end': 45},
    ]
    
    print(f"\n🔍 原始实体数: {len(raw_entities)}")
    
    # 使用 WordAligner 对齐
    aligner = WordAligner(fuzzy_threshold=0.75)
    aligned_entities = aligner.align_entities(
        entities=raw_entities,
        source_text=source_text,
        chunks=chunks,
        chunk_offsets=chunk_offsets
    )
    
    print(f"✅ 对齐后实体数: {len(aligned_entities)}")
    
    # 显示对齐报告
    report = format_alignment_report(aligned_entities, source_text, max_display=10)
    print("\n" + report)
    
    # 验证对齐结果
    print("\n🔍 验证对齐结果:")
    for entity in aligned_entities:
        actual_text = source_text[entity.start:entity.end]
        match = entity.name == actual_text
        icon = "✓" if match else "✗"
        print(f"  {icon} {entity.name}: '{actual_text}' [{entity.status.value}]")
    
    return aligned_entities


def test_fuzzy_matching():
    """测试模糊匹配功能"""
    print("\n" + "=" * 70)
    print("测试 2: 模糊匹配功能")
    print("=" * 70)
    
    # 原文（包含一些变体）
    source_text = """桂枝汤方：桂枝三两，芍药三两，甘草二两，生姜三两，大枣十二枚。
右五味，以水七升，微火煮取三升，去滓，适寒温，服一升。
"""
    
    chunks = [
        "桂枝汤方：桂枝三两，芍药三两，甘草二两，生姜三两，大枣十二枚。",
        "右五味，以水七升，微火煮取三升，去滓，适寒温，服一升。"
    ]
    
    chunk_offsets = calculate_chunk_offsets(source_text, chunks)
    
    # 模拟提取的实体（包含一些与原文略有差异的）
    raw_entities = [
        {'name': '桂枝汤', 'type': 'Prescription', 'chunk_index': 0, 'start': 0, 'end': 3},
        {'name': '桂枝', 'type': 'Herb', 'chunk_index': 0, 'start': 6, 'end': 8},
        {'name': '芍药', 'type': 'Herb', 'chunk_index': 0, 'start': 13, 'end': 15},
        {'name': '甘草', 'type': 'Herb', 'chunk_index': 0, 'start': 20, 'end': 22},
        {'name': '大枣', 'type': 'Herb', 'chunk_index': 0, 'start': 35, 'end': 37},
    ]
    
    print(f"\n📄 原文长度: {len(source_text)} 字符")
    print(f"🔍 原始实体数: {len(raw_entities)}")
    
    aligner = WordAligner(fuzzy_threshold=0.75)
    aligned_entities = aligner.align_entities(
        entities=raw_entities,
        source_text=source_text,
        chunks=chunks,
        chunk_offsets=chunk_offsets
    )
    
    print(f"✅ 对齐后实体数: {len(aligned_entities)}")
    
    # 显示对齐报告
    report = format_alignment_report(aligned_entities, source_text)
    print("\n" + report)
    
    return aligned_entities


def test_entity_extractor_integration():
    """测试与实体抽取器集成"""
    print("\n" + "=" * 70)
    print("测试 3: 与实体抽取器集成")
    print("=" * 70)
    
    # 测试文本
    text = """张三在北京大学工作，他研究人工智能。
李四在清华大学读书，专业是计算机科学。
王五在阿里巴巴担任工程师，负责云计算平台。"""
    
    # 分块
    chunks = [
        "张三在北京大学工作，他研究人工智能。",
        "李四在清华大学读书，专业是计算机科学。",
        "王五在阿里巴巴担任工程师，负责云计算平台。"
    ]
    
    chunk_offsets = calculate_chunk_offsets(text, chunks)
    
    # 使用实体抽取器
    extractor = LightweightEntityExtractor()
    
    print(f"\n📄 原文:\n{text}")
    print(f"\n📦 分块数: {len(chunks)}")
    
    # 从每个块中提取实体
    all_raw_entities = []
    for i, chunk in enumerate(chunks):
        entities = extractor.extract(chunk)
        print(f"\n  块 {i+1}: 找到 {len(entities)} 个实体")
        
        for e in entities:
            all_raw_entities.append({
                'name': e.name,
                'type': e.type,
                'chunk_index': i,
                'start': e.start,
                'end': e.end
            })
            print(f"    - {e.name} ({e.type}) @ {e.start}-{e.end}")
    
    print(f"\n🔍 总实体数: {len(all_raw_entities)}")
    
    # 使用 WordAligner 对齐
    aligner = WordAligner(fuzzy_threshold=0.75)
    aligned_entities = aligner.align_entities(
        entities=all_raw_entities,
        source_text=text,
        chunks=chunks,
        chunk_offsets=chunk_offsets
    )
    
    print(f"✅ 对齐后实体数: {len(aligned_entities)}")
    
    # 显示对齐报告
    report = format_alignment_report(aligned_entities, text)
    print("\n" + report)
    
    # 显示高亮上下文
    print("\n📝 实体高亮上下文:")
    for entity in aligned_entities[:3]:
        context = aligner.get_extraction_context(entity, text, context_chars=20)
        print(f"\n  {entity.name}:")
        print(f"    {context}")
    
    return aligned_entities


def test_verification():
    """测试验证功能"""
    print("\n" + "=" * 70)
    print("测试 4: 验证功能")
    print("=" * 70)
    
    source_text = "太阳病，发热汗出，恶风，脉缓者，名为中风。"
    
    # 创建一个对齐的实体
    aligned_entity = AlignedEntity(
        name="太阳病",
        entity_type="MedicalTerm",
        start=0,
        end=3,
        status=AlignmentStatus.MATCH_EXACT,
        similarity=1.0,
        chunk_index=0,
        chunk_relative_start=0,
        chunk_relative_end=3
    )
    
    aligner = WordAligner()
    
    print(f"\n📄 原文: {source_text}")
    print(f"\n🔍 验证实体: {aligned_entity.name}")
    
    # 验证
    verification = aligner.verify_extraction(aligned_entity, source_text)
    
    print(f"\n📊 验证结果:")
    print(f"  实体名称: {verification['entity_name']}")
    print(f"  实体类型: {verification['entity_type']}")
    print(f"  位置: {verification['position']}")
    print(f"  提取文本: '{verification['extracted_text']}'")
    print(f"  实际文本: '{verification['actual_text']}'")
    print(f"  是否匹配: {verification['match']}")
    print(f"  相似度: {verification['similarity']:.2f}")
    print(f"  对齐状态: {verification['status']}")
    print(f"  验证通过: {verification['verification_passed']}")
    
    return verification


def test_performance():
    """测试性能"""
    print("\n" + "=" * 70)
    print("测试 5: 性能测试")
    print("=" * 70)
    
    import time
    
    # 生成长文本
    base_text = "太阳病，发热汗出，恶风，脉缓者，名为中风。"
    long_text = base_text * 100  # 100 倍长度
    
    # 分块
    chunk_size = 200
    chunks = [long_text[i:i+chunk_size] for i in range(0, len(long_text), chunk_size)]
    chunk_offsets = calculate_chunk_offsets(long_text, chunks)
    
    # 生成实体
    raw_entities = []
    for i, chunk in enumerate(chunks):
        raw_entities.append({
            'name': '太阳病',
            'type': 'MedicalTerm',
            'chunk_index': i,
            'start': 0,
            'end': 3
        })
    
    print(f"\n📄 文本长度: {len(long_text)} 字符")
    print(f"📦 分块数: {len(chunks)}")
    print(f"🔍 实体数: {len(raw_entities)}")
    
    # 测试对齐性能
    aligner = WordAligner(fuzzy_threshold=0.75)
    
    start_time = time.time()
    aligned_entities = aligner.align_entities(
        entities=raw_entities,
        source_text=long_text,
        chunks=chunks,
        chunk_offsets=chunk_offsets
    )
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  对齐耗时: {elapsed:.3f}s")
    print(f"✅ 对齐实体数: {len(aligned_entities)}")
    print(f"📊 平均每个实体: {elapsed/len(raw_entities)*1000:.2f}ms")
    
    return elapsed


def main():
    """主函数"""
    print("\n" + "🧪" * 35)
    print("  WordAligner 集成测试")
    print("🧪" * 35 + "\n")
    
    try:
        # 运行所有测试
        test_basic_alignment()
        test_fuzzy_matching()
        test_entity_extractor_integration()
        test_verification()
        test_performance()
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
