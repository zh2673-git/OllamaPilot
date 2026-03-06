# SearXNG 配置脚本
# 在容器启动后运行此脚本配置 JSON 输出和搜索引擎

# 检查容器是否存在
$container = docker ps -aq -f name=searxng
if (-not $container) {
    Write-Host "❌ 容器不存在，请先运行: docker-compose up -d"
    exit 1
}

# 检查容器是否运行
$running = docker ps -q -f name=searxng
if (-not $running) {
    Write-Host "启动容器..."
    docker-compose start
}

Write-Host "等待 SearXNG 启动..."
Start-Sleep 5

Write-Host "配置 SearXNG..."

# 启用 JSON 格式
docker exec searxng sed -i '77,78s/.*/  formats: [html, json]/' /etc/searxng/settings.yml 2>$null
docker exec searxng sed -i '78d' /etc/searxng/settings.yml 2>$null

# 启用搜索引擎
docker exec searxng sed -i 's/disabled: true/disabled: false/g' /etc/searxng/settings.yml 2>$null

# 重启容器应用配置
docker restart searxng

Write-Host "等待 SearXNG 重新启动..."
Start-Sleep 5

# 测试
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/search?q=test&format=json" -UseBasicParsing -TimeoutSec 10
    $data = $response.Content | ConvertFrom-Json
    Write-Host "✅ SearXNG 配置成功！找到 $($data.results.Count) 个结果"
    Write-Host "`n💡 提示: 只要容器不被删除，配置就会一直保留"
    Write-Host "   停止: docker-compose stop"
    Write-Host "   启动: docker-compose start"
} catch {
    Write-Host "❌ 测试失败: $_"
}

Write-Host "`nSearXNG 已配置完成，访问: http://localhost:8080"
