"""
测试工具过滤功能
"""
import asyncio
from ollamapilot import init_ollama_model
from ollamapilot.agent import OllamaPilotAgent

async def test():
    print("🔄 初始化模型...")
    model = init_ollama_model()
    
    print("🔄 创建 Agent...")
    agent = OllamaPilotAgent(
        model=model,
        verbose=True,
        enable_memory=True,
    )
    
    print("\n📝 测试查询: 今天的国际新闻有哪些")
    print("=" * 50)
    
    full_response = ""
    async for event in agent.astream_events("今天的国际新闻有哪些", thread_id="test"):
        event_type = event.get("event", "")
        
        if event_type == "on_chat_model_stream":
            data = event.get("data", {})
            chunk = data.get("chunk", None)
            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
    
    print("\n" + "=" * 50)
    print(f"✅ 测试完成，回复长度: {len(full_response)} 字符")

if __name__ == "__main__":
    asyncio.run(test())
